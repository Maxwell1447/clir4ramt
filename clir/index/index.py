"""Indexing script

use python -m clir.index.index --config experiments/configs/index.yaml
"""
import argparse
import yaml
import os
import torch
from ..train.trainee import BiEncoder
from ..data.data import load_monolingual_corpus
from transformers import BertModel


def indexerCli(data=None, checkpoint=None, model_kwargs=None, device=None, index_dir=None, index_name=None, **kwargs):
    """
    data_path: str
        path to data to index
    checkpoint: str
        path to bi-Encoder checkpoint
    """
    if device != "cpu":
        assert torch.cuda.is_available(), "No cuda device found"
    module = BiEncoder.load_from_checkpoint(checkpoint, **model_kwargs)
    model = module.tgt_model.to(device)
    del module
    dataloader, dic = load_monolingual_corpus(**data)
    outs = list()
    with torch.no_grad():
        for samples in iter(dataloader):
            tokens = samples["tokens"].to(device)
            outs.append(model(
                input_ids=tokens,
                attention_mask=tokens.ne(dic.pad())
            ).last_hidden_state[:, 0, :].cpu())
    outs = torch.cat(outs)
    if not os.path.isdir(index_dir):
        os.mkdir(index_dir)
    torch.save(outs, os.path.join(index_dir, index_name))
    


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--config", required=True, help="YAML file path")
    return args.parse_args()

def main():
    args = parser()
    assert os.path.exists(args.config)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return indexerCli(**config)

if __name__ == "__main__":
    main()