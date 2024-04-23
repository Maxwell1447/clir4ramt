"""Indexing script

use python -m clir.index.index --config experiments/configs/index.yaml
"""
import argparse
import yaml
import os
import torch
import numpy as np
from ..train.trainee import BiEncoder
from ..data.data import load_monolingual_corpus
import faiss


def search(query, index, device, not_same_index, k):
    if not_same_index:
        D, I = index.search(query, k + 1)
        # exclude same index match
        I = I[
            np.arange(len(I))[:, np.newaxis],
            np.argsort(I == np.arange(len(I))[:, np.newaxis], kind='stable')
        ][:, :-1]
    else:
        D, I = index.search(query, k)
    return I


def retrieve_index_to_index(index_source, index_path, device, not_same_index, k):
    d_src = torch.load(index_source, map_location=torch.device("cpu")).numpy()
    d_tgt = torch.load(index_path, map_location=torch.device("cpu")).numpy()
    index = faiss.IndexFlatIP(d_tgt.shape[1])
    index.add(d_tgt)
    return search(d_src, index, device, not_same_index, k)


def retrieve_source_to_index(
    data=None,
    checkpoint=None,
    model_kwargs=None, index_path=None, device=None, not_same_index=None, k=None
):
    d_tgt = torch.load(index_path, map_location=torch.device("cpu")).numpy()
    index = faiss.IndexFlatIP(d_tgt.shape[1])
    index.add(d_tgt)
    module = BiEncoder.load_from_checkpoint(checkpoint, **model_kwargs)
    model = module.src_model.to(device)
    del module
    dataloader, dic = load_monolingual_corpus(**data)
    outs = list()
    with torch.no_grad():
        for samples in iter(dataloader):
            tokens = samples["tokens"].to(device)
            outs.append(
                search(
                    model(
                        input_ids=tokens,
                        attention_mask=tokens.ne(dic.pad())
                    ).last_hidden_state[:, 0, :].cpu().contiguous().numpy(),
                    index,
                    device,
                    not_same_index,
                    k
                )
            )
            break
    return np.concatenate(outs)

def retrieverCli(data=None, checkpoint=None, model_kwargs=None, device="cpu", index_path=None, index_source=None, not_same_index=False, k=1, save_path=None, **kwargs):
    """
    data_path: str
        path to data to index
    checkpoint: str
        path to bi-Encoder checkpoint
    """
    print(checkpoint)
    if device != "cpu":
        assert torch.cuda.is_available(), "No cuda device found"
    if index_source is not None and os.path.exists(index_source):
        out = retrieve_index_to_index(index_source, index_path, device, not_same_index, k)
    else:
        out = retrieve_source_to_index(data=data, model_kwargs=model_kwargs, checkpoint=checkpoint, index_path=index_path, device=device, not_same_index=not_same_index, k=k)
    # print(out[:5])
    if save_path is not None and not os.path.isdir(save_path):
        os.mkdir(save_path)

    np.save(os.path.join(save_path, "indices.npy"), out)
    


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--config", required=True, help="YAML file path")
    return args.parse_args()

def main():
    args = parser()
    assert os.path.exists(args.config)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return retrieverCli(**config)

if __name__ == "__main__":
    main()