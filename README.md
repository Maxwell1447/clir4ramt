# Cross-Lingual Information Retrieval for Neural Machien Translation

## installation requirements

Fairseq requires to be installed and is used for data handling.

```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install .
python setup build_ext --inplace
```

To install this package:
```bash
git clone https://github.com/Maxwell1447/clir4ramt.git
cd clir
pip install .
```

```bash
python -m clir.train.trainer fit --config=experiments/configs/{config}.yaml
```

```bash
python -m clir.index.index --config experiments/configs/{config}.yaml
```