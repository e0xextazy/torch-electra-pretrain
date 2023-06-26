from transformers import ElectraTokenizerFast
import datasets

hf_tokenizer = ElectraTokenizerFast.from_pretrained(
    f"google/electra-large-discriminator", do_lower_case=False)


def get_training_corpus():
    dataset = datasets.load_dataset(
        'text', data_files="train_data_clean.txt", cache_dir='./datasets')["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx: start_idx + 1000]
        yield samples["text"]


training_corpus = get_training_corpus()

new_hf_tokenizer = hf_tokenizer.train_new_from_iterator(training_corpus, 30522)
new_hf_tokenizer.save_pretrained("my_tokenizer_30522")
