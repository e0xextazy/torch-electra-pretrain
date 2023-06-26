import os
import random
from pathlib import Path
from functools import partial
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import datasets
from fastai.text.all import *
from transformers import ElectraConfig, ElectraTokenizerFast, ElectraForMaskedLM, ElectraForPreTraining
from hugdatafast import *
import tensorboard
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.distributed import *


class ELECTRADataProcessor(object):
    """Given a stream of input text, creates pretraining examples."""

    def __init__(self, hf_dset, hf_tokenizer, max_length, text_col='text', lines_delimiter='\n', minimize_data_size=True, apply_cleaning=True):
        self.hf_tokenizer = hf_tokenizer
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._target_length = max_length

        self.hf_dset = hf_dset
        self.text_col = text_col
        self.lines_delimiter = lines_delimiter
        self.minimize_data_size = minimize_data_size
        self.apply_cleaning = apply_cleaning

    def map(self, **kwargs):
        "Some settings of datasets.Dataset.map for ELECTRA data processing"
        num_proc = kwargs.pop('num_proc', os.cpu_count())
        return self.hf_dset.map(
            function=self,
            batched=True,
            # this is must b/c we will return different number of rows
            remove_columns=self.hf_dset.column_names,
            disable_nullable=False,
            input_columns=[self.text_col],
            writer_batch_size=10**4,
            num_proc=num_proc,
            **kwargs
        )

    def __call__(self, texts):
        if self.minimize_data_size:
            new_example = {'input_ids': [], 'sentA_length': []}
        else:
            new_example = {'input_ids': [],
                           'input_mask': [], 'segment_ids': []}

        for text in texts:  # for every doc

            for line in re.split(self.lines_delimiter, text):  # for every paragraph

                if re.fullmatch(r'\s*', line):
                    continue  # empty string or string with all space characters
                if self.apply_cleaning and self.filter_out(line):
                    continue

                example = self.add_line(line)
                if example:
                    for k, v in example.items():
                        new_example[k].append(v)

            if self._current_length != 0:
                example = self._create_example()
                for k, v in example.items():
                    new_example[k].append(v)

        return new_example

    def filter_out(self, line):
        if len(line) < 80:
            return True
        return False

    def clean(self, line):
        # () is remainder after link in it filtered out
        return line.strip().replace("\n", " ").replace("()", "")

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = self.clean(line)
        tokens = self.hf_tokenizer.tokenize(line)
        tokids = self.hf_tokenizer.convert_tokens_to_ids(tokens)
        self._current_sentences.append(tokids)
        self._current_length += len(tokids)
        if self._current_length >= self._target_length:
            return self._create_example()
        return None

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self._target_length - 3) // 2

        first_segment = []
        second_segment = []
        for sentence in self._current_sentences:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (len(first_segment) == 0 or len(first_segment) + len(sentence) < first_segment_target_length or (len(second_segment) == 0 and len(first_segment) < first_segment_target_length and random.random() < 0.5)):
                first_segment += sentence
            else:
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self._max_length - 2]
        second_segment = second_segment[:max(
            0, self._max_length - len(first_segment) - 3)]

        # prepare to start building the next example
        self._current_sentences = []
        self._current_length = 0
        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        return self._make_example(first_segment, second_segment)

    def _make_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a tf.train.Example."""
        input_ids = [self.hf_tokenizer.cls_token_id] + \
            first_segment + [self.hf_tokenizer.sep_token_id]
        sentA_length = len(input_ids)
        segment_ids = [0] * sentA_length
        if second_segment:
            input_ids += second_segment + [self.hf_tokenizer.sep_token_id]
            segment_ids += [1] * (len(second_segment) + 1)

        if self.minimize_data_size:
            return {
                'input_ids': input_ids,
                'sentA_length': sentA_length,
            }
        else:
            input_mask = [1] * len(input_ids)
            input_ids += [0] * (self._max_length - len(input_ids))
            input_mask += [0] * (self._max_length - len(input_mask))
            segment_ids += [0] * (self._max_length - len(segment_ids))
            return {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            }


class MyConfig(dict):
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value


def adam_no_correction_step(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, **kwargs):
    p.data.addcdiv_(grad_avg, (sqr_avg).sqrt() + eps, value=-lr)
    return p


def Adam_no_bias_correction(params, lr, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.01, decouple_wd=True):
    "A `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += [partial(average_grad, dampening=True),
            average_sqr_grad, step_stat, adam_no_correction_step]
    return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)


def linear_warmup_and_decay(pct, lr_max, total_steps, warmup_steps=None, warmup_pct=None, end_lr=0.0, decay_power=1):
    """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
    if warmup_pct:
        warmup_steps = int(warmup_pct * total_steps)
    step_i = round(pct * total_steps)
    # According to the original source code, two schedules take effect at the same time, but decaying schedule will be neglible in the early time.
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay
    decayed_lr = (lr_max-end_lr) * (1 - step_i /
                                    total_steps) ** decay_power + end_lr
    # https://github.com/google-research/electra/blob/81f7e5fc98b0ad8bfd20b641aa8bc9e6ac00c8eb/model/optimization.py#L44
    warmed_lr = decayed_lr * min(1.0, step_i/warmup_steps)
    return warmed_lr


def linear_warmup_and_then_decay(pct, lr_max, total_steps, warmup_steps=None, warmup_pct=None, end_lr=0.0, decay_power=1):
    """ pct (float): fastai count it as ith_step/num_epoch*len(dl), so we can't just use pct when our num_epoch is fake.he ith_step is count from 0, """
    if warmup_pct:
        warmup_steps = int(warmup_pct * total_steps)
    step_i = round(pct * total_steps)
    if step_i <= warmup_steps:  # warm up
        return lr_max * min(1.0, step_i/warmup_steps)
    else:  # decay
        return (lr_max-end_lr) * (1 - (step_i-warmup_steps)/(total_steps-warmup_steps)) ** decay_power + end_lr


c = MyConfig({
    'device': 'cuda',

    'base_run_name': 'vanilla_30522',  # run_name = {base_run_name}_{seed}
    # 11081 36 1188 76 1 4 4649 7 # None/False to randomly choose seed from [0,999999]
    'seed': 228,

    'adam_bias_correction': False,
    'schedule': 'original_linear',
    'sampling': 'fp32_gumbel',
    'electra_mask_style': True,
    'gen_smooth_label': False,
    'disc_smooth_label': False,

    'size': 'large',
    'datas': ['russian'],

    'logger': 'wandb',
    'num_workers': 16,
    'resume': False,
    'ckpt': ''
})

# Check and Default
assert c.sampling in ['fp32_gumbel', 'fp16_gumbel', 'multinomial']
assert c.schedule in ['original_linear',
                      'separate_linear', 'one_cycle', 'adjusted_one_cycle']
for data in c.datas:
    assert data in ['wikipedia', 'bookcorpus', 'openwebtext', 'russian']
assert c.logger in ['wandb', 'neptune', None, False]
if not c.base_run_name:
    c.base_run_name = str(datetime.now(timezone(timedelta(
        hours=+8))))[6:-13].replace(' ', '').replace(':', '').replace('-', '')
if not c.seed:
    c.seed = random.randint(0, 999999)
c.run_name = f'{c.base_run_name}_{c.seed}'
if c.gen_smooth_label is True:
    c.gen_smooth_label = 0.1
if c.disc_smooth_label is True:
    c.disc_smooth_label = 0.1

i = ['small', 'base', 'large'].index(c.size)
c.mask_prob = [0.15, 0.15, 0.25][i]
c.lr = [5e-4, 2e-4, 2e-4][i]
c.bs = [128, 256, 48][i]  # 2048
c.steps = [10**6, 766*1000, 400*1000][i]
c.max_length = [128, 512, 512][i]
generator_size_divisor = [4, 3, 4][i]

disc_config = ElectraConfig(
    vocab_size=120_138,
    max_position_embeddings=512,
    num_attention_heads=16,
    num_hidden_layers=24,  # 6,
    type_vocab_size=2,
    hidden_size=1024,
    intermediate_size=4096,
    embedding_size=1024,
)

gen_config = ElectraConfig(
    vocab_size=120_138,
    max_position_embeddings=512,
    num_attention_heads=4,
    num_hidden_layers=24,  # 6,
    type_vocab_size=2,
    hidden_size=256,
    intermediate_size=1024,
    embedding_size=1024,
)

hf_tokenizer = ElectraTokenizerFast.from_pretrained('my_tokenizer_30522')
rank0_first(print, 'Vocab size: ', hf_tokenizer.vocab_size)

Path('./datasets').mkdir(exist_ok=True)
Path('./checkpoints/pretrain').mkdir(exist_ok=True, parents=True)
edl_cache_dir = Path("./datasets/electra_dataloader")
edl_cache_dir.mkdir(exist_ok=True)

rank0_first(print, f"process id: {os.getpid()}")
rank0_first(print, c)

dsets = []
ELECTRAProcessor = partial(
    ELECTRADataProcessor, hf_tokenizer=hf_tokenizer, max_length=c.max_length)


def get_dataset():
    print('load/download russian dataset')
    russian = datasets.load_dataset(
        'text', data_files="train_data.txt", cache_dir='./datasets')["train"]
    print('load/create data from russian dataset for ELECTRA')
    e_russian = ELECTRAProcessor(russian).map(
        cache_file_name=f"electra_russian_{c.max_length}.arrow", num_proc=64)

    return e_russian


if 'russian' in c.datas:
    e_russian = rank0_first(get_dataset)
    dsets.append(e_russian)

assert len(dsets) == len(c.datas)

merged_dsets = {'train': datasets.concatenate_datasets(dsets)}
hf_dsets = HF_Datasets(
    merged_dsets,
    cols={'input_ids': TensorText, 'sentA_length': noop},
    hf_toker=hf_tokenizer,
    n_inp=2,
)

dls = hf_dsets.dataloaders(
    bs=c.bs,
    num_workers=c.num_workers,
    pin_memory=False,
    shuffle_train=True,
    srtkey_fc=False,
    cache_dir='./datasets/electra_dataloader',
    cache_name='dl_{split}.json'
)


def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1, orginal_prob=0.1, ignore_index=-100):
    """ 
    Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK, replace_prob% random, orginal_prob% original within mlm_probability% of tokens in the sentence. 
    * ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss
    """

    device = inputs.device
    labels = inputs.clone()

    # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
    probability_matrix = torch.full(
        labels.shape, mlm_probability, device=device)
    special_tokens_mask = torch.full(
        inputs.shape, False, dtype=torch.bool, device=device)
    for sp_id in special_token_indices:
        special_tokens_mask = special_tokens_mask | (inputs == sp_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    mlm_mask = torch.bernoulli(probability_matrix).bool()
    # We only compute loss on mlm applied tokens
    labels[~mlm_mask] = ignore_index

    # mask  (mlm_probability * (1-replace_prob-orginal_prob))
    mask_prob = 1 - replace_prob - orginal_prob
    mask_token_mask = torch.bernoulli(torch.full(
        labels.shape, mask_prob, device=device)).bool() & mlm_mask
    inputs[mask_token_mask] = mask_token_index

    # replace with a random token (mlm_probability * replace_prob)
    if int(replace_prob) != 0:
        rep_prob = replace_prob/(replace_prob + orginal_prob)
        replace_token_mask = torch.bernoulli(torch.full(
            labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
        random_words = torch.randint(
            vocab_size, labels.shape, dtype=torch.long, device=device)
        inputs[replace_token_mask] = random_words[replace_token_mask]

    # do nothing (mlm_probability * orginal_prob)
    pass

    return inputs, labels, mlm_mask


class MaskedLMCallback(Callback):
    @delegates(mask_tokens)
    def __init__(self, mask_tok_id, special_tok_ids, vocab_size, ignore_index=-100, for_electra=False, **kwargs):
        self.ignore_index = ignore_index
        self.for_electra = for_electra
        self.mask_tokens = partial(mask_tokens,
                                   mask_token_index=mask_tok_id,
                                   special_token_indices=special_tok_ids,
                                   vocab_size=vocab_size,
                                   ignore_index=-100,
                                   **kwargs)

    def before_batch(self):
        input_ids, sentA_lenths = self.xb
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(input_ids)
        if self.for_electra:
            self.learn.xb, self.learn.yb = (
                masked_inputs, sentA_lenths, is_mlm_applied, labels), (labels,)
        else:
            self.learn.xb, self.learn.yb = (
                masked_inputs, sentA_lenths), (labels,)

    @delegates(TfmdDL.show_batch)
    def show_batch(self, dl, idx_show_ignored, verbose=True, **kwargs):
        b = dl.one_batch()
        input_ids, sentA_lenths = b
        masked_inputs, labels, is_mlm_applied = self.mask_tokens(
            input_ids.clone())
        # check
        assert torch.equal(is_mlm_applied, labels != self.ignore_index)
        assert torch.equal((~is_mlm_applied * masked_inputs +
                           is_mlm_applied * labels), input_ids)
        # change symbol to show the ignored position
        labels[labels == self.ignore_index] = idx_show_ignored
        # some notice to help understand the masking mechanism
        if verbose:
            print("We won't count loss from position where y is ignore index")
            print(
                "Notice 1. Positions have label token in y will be either [Mask]/other token/orginal token in x")
            print("Notice 2. Special tokens (CLS, SEP) won't be masked.")
            print(
                "Notice 3. Dynamic masking: every time you run gives you different results.")
        # show
        tfm_b = (masked_inputs, sentA_lenths, is_mlm_applied,
                 labels) if self.for_electra else (masked_inputs, sentA_lenths, labels)
        dl.show_batch(b=tfm_b, **kwargs)


mlm_cb = MaskedLMCallback(mask_tok_id=hf_tokenizer.mask_token_id,
                          special_tok_ids=hf_tokenizer.all_special_ids,
                          vocab_size=hf_tokenizer.vocab_size,
                          mlm_probability=c.mask_prob,
                          replace_prob=0.0 if c.electra_mask_style else 0.1,
                          orginal_prob=0.15 if c.electra_mask_style else 0.1,
                          for_electra=True)


class ELECTRAModel(nn.Module):
    def __init__(self, generator, discriminator, hf_tokenizer):
        super().__init__()
        self.generator, self.discriminator = generator, discriminator
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0., 1.)
        self.hf_tokenizer = hf_tokenizer

    def to(self, *args, **kwargs):
        "Also set dtype and device of contained gumbel distribution if needed"
        super().to(*args, **kwargs)
        a_tensor = next(self.parameters())
        device, dtype = a_tensor.device, a_tensor.dtype
        if c.sampling == 'fp32_gumbel':
            dtype = torch.float32
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(
            0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

    def forward(self, masked_inputs, sentA_lenths, is_mlm_applied, labels):
        """
        masked_inputs (Tensor[int]): (B, L)
        sentA_lenths (Tensor[int]): (B, L)
        is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability 
        labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
        """
        attention_mask, token_type_ids = self._get_pad_mask_and_token_type(
            masked_inputs, sentA_lenths)
        gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[
            0]  # (B, L, vocab size)
        # reduce size to save space and speed
        # ( #mlm_positions, vocab_size)
        mlm_gen_logits = gen_logits[is_mlm_applied, :]

        with torch.no_grad():
            # sampling
            pred_toks = self.sample(mlm_gen_logits)  # ( #mlm_positions, )
            # produce inputs for discriminator
            generated = masked_inputs.clone()  # (B,L)
            generated[is_mlm_applied] = pred_toks  # (B,L)
            # produce labels for discriminator
            is_replaced = is_mlm_applied.clone()  # (B,L)
            is_replaced[is_mlm_applied] = (
                pred_toks != labels[is_mlm_applied])  # (B,L)

        disc_logits = self.discriminator(
            generated, attention_mask, token_type_ids)[0]  # (B, L)

        return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

    def _get_pad_mask_and_token_type(self, input_ids, sentA_lenths):
        """
        Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
        """
        attention_mask = input_ids != self.hf_tokenizer.pad_token_id
        seq_len = input_ids.shape[1]
        token_type_ids = torch.tensor([([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],
                                      device=input_ids.device)
        return attention_mask, token_type_ids

    def sample(self, logits):
        "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"
        if c.sampling == 'fp32_gumbel':
            gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
            return (logits.float() + gumbel).argmax(dim=-1)
        elif c.sampling == 'fp16_gumbel':  # 5.06 ms
            gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
            return (logits + gumbel).argmax(dim=-1)
        elif c.sampling == 'multinomial':  # 2.X ms
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()


class ELECTRALoss():
    def __init__(self, loss_weights=(1.0, 50.0), gen_label_smooth=False, disc_label_smooth=False):
        self.loss_weights = loss_weights
        self.gen_loss_fc = LabelSmoothingCrossEntropyFlat(
            eps=gen_label_smooth) if gen_label_smooth else CrossEntropyLossFlat()
        self.disc_loss_fc = nn.BCEWithLogitsLoss()
        self.disc_label_smooth = disc_label_smooth

    def __call__(self, pred, targ_ids):
        mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
        gen_loss = self.gen_loss_fc(
            mlm_gen_logits.float(), targ_ids[is_mlm_applied])
        disc_logits = disc_logits.masked_select(non_pad)  # -> 1d tensor
        is_replaced = is_replaced.masked_select(non_pad)  # -> 1d tensor
        if self.disc_label_smooth:
            is_replaced = is_replaced.float().masked_fill(
                ~is_replaced, self.disc_label_smooth)
        disc_loss = self.disc_loss_fc(disc_logits.float(), is_replaced.float())
        return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]


torch.backends.cudnn.benchmark = True
dls[0].rng = random.Random(c.seed)  # for fastai dataloader
random.seed(c.seed)
np.random.seed(c.seed)
torch.manual_seed(c.seed)

generator = ElectraForMaskedLM(gen_config)
discriminator = ElectraForPreTraining(disc_config)
discriminator.electra.embeddings = generator.electra.embeddings
generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight

electra_model = ELECTRAModel(generator, discriminator, hf_tokenizer)

electra_model = torch.nn.DataParallel(electra_model, device_ids=[0, 1, 2, 3])
electra_loss_func = ELECTRALoss(
    gen_label_smooth=c.gen_smooth_label, disc_label_smooth=c.disc_smooth_label)

if c.adam_bias_correction:
    opt_func = partial(Adam, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=0.01)
else:
    opt_func = partial(Adam_no_bias_correction, eps=1e-6,
                       mom=0.9, sqr_mom=0.999, wd=0.01)

if c.schedule.endswith('linear'):
    lr_shed_func = linear_warmup_and_then_decay if c.schedule == 'separate_linear' else linear_warmup_and_decay
    lr_shedule = ParamScheduler({'lr': partial(lr_shed_func,
                                               lr_max=c.lr,
                                               warmup_steps=10000,
                                               total_steps=c.steps,)})

# Learner


class RunSteps(Callback):
    toward_end = True

    def __init__(self, n_steps, save_points=None, base_name=None, no_val=True):
        """
        Args:
        `n_steps` (`Int`): Run how many steps, could be larger or smaller than `len(dls.train)`
        `savepoints` 
        - (`List[Float]`): save when reach one of percent specified.
        - (`List[Int]`): save when reache one of steps specified
        `base_name` (`String`): a format string with `{percent}` to be passed to `learn.save`.
        """
        if save_points is None:
            save_points = []
        else:
            assert '{percent}' in base_name
            save_points = [s if isinstance(s, int) else int(
                n_steps*s) for s in save_points]
            for sp in save_points:
                assert sp != 1, "Are you sure you want to save after 1 steps, instead of 1.0 * num_steps ?"
            assert max(save_points) <= n_steps
        store_attr('n_steps,save_points,base_name,no_val', self)

    def before_train(self):
        # fix pct_train (cuz we'll set `n_epoch` larger than we need)
        self.learn.pct_train = self.train_iter/self.n_steps

    def after_batch(self):
        # fix pct_train (cuz we'll set `n_epoch` larger than we need)
        self.learn.pct_train = self.train_iter/self.n_steps
        # when to save
        if self.train_iter in self.save_points:
            percent = (self.train_iter/self.n_steps)*100
            self.learn.save(self.base_name.format(percent=f'{percent}%'))
        # when to interrupt
        if self.train_iter == self.n_steps:
            raise CancelFitException

    def after_train(self):
        if self.no_val:
            if self.train_iter == self.n_steps:
                pass  # CancelFit is raised, don't overlap it with CancelEpoch
            else:
                raise CancelEpochException


dls.to(torch.device(c.device))
learn = Learner(dls, electra_model,
                loss_func=electra_loss_func,
                opt_func=opt_func,
                path='./checkpoints',
                model_dir='pretrain',
                cbs=[mlm_cb,
                     # every 10k steps
                     RunSteps(
                         c.steps, [0.025*i for i in range(1, 41)], c.run_name+"_{percent}"),
                     ],
                )

if c.resume:
    learn.load(c.ckpt)
    print("checkpoint was uploaded!!!")

hparam_update = {

}

learn.add_cb(TensorBoardCallback(trace_model=False))


class GradientClipping(Callback):
    def __init__(self, clip: float = 0.1):
        self.clip = clip
        assert self.clip

    def after_backward(self):
        if hasattr(self, 'scaler'):
            self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)


# Mixed precison and Gradient clip
learn.to_fp16(init_scale=2.**11)
learn.add_cb(GradientClipping(1.))

# Print time and run name
print(f"{c.run_name} , starts at {datetime.now()}")

# Run
if c.schedule == 'one_cycle':
    learn.fit_one_cycle(9999, lr_max=c.lr)
elif c.schedule == 'adjusted_one_cycle':
    learn.fit_one_cycle(9999, lr_max=c.lr, div=1e5, pct_start=10000/c.steps)
else:
    learn.fit(9999, cbs=[lr_shedule])


# 2 gpu 13500, 8500 | 24k time, 17kk steps, bs 12
# 4 gpu 11200, 6300, 6300, 6300 | 28k time, 17kk steps, bs12
# 4 gpu 22000, 17000, 17000, 17000 | 13k time, 6.4kk steps, bs32
# 4 gpu 27600, 23500, 23500, 23500 | 8k time, 4.2kk steps, bs48
# 1 gpu 27400 | 25k time, 17kk steps, bs 12
# CUDA_VISIBLE_DEVICES="2,4,6,7" nohup python pretrain.py > logs.out &
