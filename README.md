Reference:
> https://github.com/richarddwang/electra_pytorch by Richard Wang

# Unofficial PyTorch implementation of
> [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) by Kevin Clark. Minh-Thang Luong. Quoc V. Le. Christopher D. Manning

Updates:
- Add multi GPU DataParallel

## Usage
1. `git clone` and `pip install requirements.txt`
2. Train your tokenizer: `python tokenizer_train.py`
3. Start pretrain: `CUDA_VISIBLE_DEVICES="2,4,6,7" python pretrain.py`
