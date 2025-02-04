# Improving Retrieval-Augmented Neural Machine Translation with Monolingual Data​

CLIR4NMT: Cross-Lingual Information Retrieval for Retrieval Augmented Machine Translation

## installation requirements

Fairseq requires to be installed and is used for data handling.

```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install .
python setup.py build_ext --inplace
```

To install this package:
```bash
git clone https://github.com/Maxwell1447/clir4ramt.git
cd clir
pip install .
```

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## training

```bash
python -m clir.train.trainer fit --config=experiments/configs/{config}.yaml
```

## indexing

```bash
python -m clir.index.index --config experiments/configs/{config}.yaml
```

## retrieving

```bash
python -m clir.retrieve.retrieve --config experiments/configs/{config}.yaml
```

## retrieval
