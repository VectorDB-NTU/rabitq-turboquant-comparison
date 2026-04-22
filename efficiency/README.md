# Efficiency

This folder corresponds to the quantization efficiency experiments for RaBitQ in the paper.

## Structure
- `cpu/`: CPU-based efficiency experiments
- `gpu/`: GPU-based efficiency experiments

## Data Preparation

Download 3 datasets:

```bash
python get_dataset.py --dim 1536
python get_dataset.py --dim 3072
wget https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.200d.zip
unzip glove.2024.wikigiga.200d.zip
python prepare_glove200.py
```