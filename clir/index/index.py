"""Indexing script

use python -m clir.index.index --config experiments/configs/index.yaml
"""
import argparse
import yaml
import os
import sys
import torch
import torch.nn as nn
from ..train.trainee import BiEncoder
from ..data.data import load_monolingual_corpus
from tqdm import tqdm
# from transformers import BertModel
import numpy as np
import gzip


def indexerCli(data=None, checkpoint=None, side="target", model_kwargs=None, device="cpu", index_dir=None, index_name=None, **kwargs):
    """
    data_path: str
        path to data to index
    checkpoint: str
        path to bi-Encoder checkpoint
    """
    if device != "cpu":
        assert torch.cuda.is_available(), "No cuda device found"
    module = BiEncoder.load_from_checkpoint(checkpoint, **model_kwargs)
    if side != "target":
        model = module.src_model.to(device)
    else:
        model = module.tgt_model.to(device)
    model.eval()
    del module
    dataloader, dic = load_monolingual_corpus(**data)
    # data_itr = iter(dataloader)
    data_itr = dataloader.next_epoch_itr(shuffle=False)
    outs = list()
    ids = list()
    with torch.no_grad():
        for samples in tqdm(data_itr):
            tokens = samples["tokens"].to(device)
            outs.append(nn.functional.normalize(model(
                input_ids=tokens,
                attention_mask=tokens.ne(dic.pad())
            ).last_hidden_state[:, 0, :]).half().cpu())
            ids.append(samples["id"])
            assert len(ids[-1]) == outs[-1].shape[0], f"{len(ids[-1])} != {outs[-1].shape[0]}"
    #         print(tokens)
    #         break
    # sys.exit(8)
    outs = torch.cat(outs)
    ids = torch.cat(ids)
    outs_ord = torch.empty_like(outs)
    outs_ord.scatter_(0, ids.view(-1, 1).expand_as(outs), outs)

    if not os.path.isdir(index_dir):
        os.mkdir(index_dir)
    np.save(f"{os.path.join(index_dir, index_name)}", outs_ord.numpy())

    


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--config", required=True, help="YAML file path")
    args.add_argument("--name", default="ALL", help="domain dataset")
    return args.parse_args()

def main():
    args = parser()
    assert os.path.exists(args.config)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.name != "ALL":
        config["data"]["data_path"] = os.path.join(
            os.path.dirname(config["data"]["data_path"]),
            args.name,
            os.path.basename(config["data"]["data_path"]))
        config["index_dir"] = os.path.join(
            config["index_dir"],
            args.name)
        if not os.path.isdir(config["index_dir"]):
            os.mkdir(config["index_dir"])

    return indexerCli(**config)

if __name__ == "__main__":
    main()