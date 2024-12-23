"""Indexing script

use python -m clir.index.index --config experiments/configs/index.yaml
"""
import argparse
import yaml
import os
import sys
import torch
import numpy as np
from ..train.trainee import BiEncoder
from ..data.data import load_monolingual_corpus
import faiss
import gzip
import time

def search(query, index, device, not_same_index, k):
    # print(not_same_index)
    # sys.exit(8)
    print("searching...")
    if not_same_index:
        D, I = index.search(query, k + 1)
        # exclude same index match
        I = I[
            np.arange(len(I))[:, np.newaxis],
            np.argsort(I == np.arange(len(I))[:, np.newaxis], kind='stable')
        ][:, :-1]
        D = D[
            np.arange(len(I))[:, np.newaxis],
            np.argsort(I == np.arange(len(I))[:, np.newaxis], kind='stable')
        ][:, :-1]
    else:
        D, I = index.search(query, k)
    print("...done")

    return D, I


def retrieve_index_to_index(index_source, index_path, device, not_same_index, k, ivf):
    # with gzip.GzipFile(index_source, 'r') as f:
    #     d_src = np.load(f).astype(np.float32)
    # with gzip.GzipFile(index_path, 'r') as f:
    #     d_tgt = np.load(f).astype(np.float32)
    d_src = np.load(index_source).astype(np.float32)
    d_tgt = np.load(index_path).astype(np.float32)
    ####### LOW MEMORY
    # N = len(d_src)
    # d_src = d_src[:N//2]
    # d_tgt = d_tgt[:N//2]
    #######
    # print((d_src == d_tgt).all(-1).sum()/ len(d_src))
    # d_src = np.random.rand(100, 128).astype(np.float32)
    # d_tgt = np.random.rand(1000, 128).astype(np.float32)
    # d_tgt = d_src
    # print(d_src)
    # d_src = torch.load(index_source, map_location=torch.device("cpu")).numpy()
    # d_tgt = torch.load(index_path, map_location=torch.device("cpu")).numpy()
    print(d_src.shape, d_tgt.shape)
    t1 = time.time()
    index = faiss.IndexFlatIP(d_tgt.shape[1])
    if ivf:
        index = faiss.IndexIVFFlat(index, d_tgt.shape[1], 100)
        index.train(d_tgt)
    if device != "cpu":
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    index.add(d_tgt)
    t2 = time.time()
    print("indexing:", t2 - t1)
    scores, res = search(d_src, index, device, not_same_index, k)
    t3 = time.time()
    print("search:", t3 - t2)
    return scores, res


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
    scores = list()
    with torch.no_grad():
        for samples in iter(dataloader):
            tokens = samples["tokens"].to(device)
            score, ids = search(
                model(
                    input_ids=tokens,
                    attention_mask=tokens.ne(dic.pad())
                ).last_hidden_state[:, 0, :].cpu().contiguous().numpy(),
                index,
                device,
                not_same_index,
                k
            )
            outs.append(ids)
            scores.append(score)
            break
    return np.concatenate(scores), np.concatenate(outs)

def retrieverCli(data=None, checkpoint=None, model_kwargs=None, device="cpu", index_path=None, index_source=None, not_same_index=False, k=1, ivf=False, save_path=None, save_name="-", **kwargs):
    """
    data_path: str
        path to data to index
    checkpoint: str
        path to bi-Encoder checkpoint
    """
    print(checkpoint)
    if device != "cpu":
        assert torch.cuda.is_available(), "No cuda device found"
    print(os.path.exists(index_source), index_source)
    if index_source is not None and os.path.exists(index_source):
        scores, ids = retrieve_index_to_index(index_source, index_path, device, not_same_index, k, ivf)
    else:
        scores, ids = retrieve_source_to_index(data=data, model_kwargs=model_kwargs, checkpoint=checkpoint, index_path=index_path, device=device, not_same_index=not_same_index, k=k)
    # print(out[:5])
    if save_path is not None and not os.path.isdir(save_path):
        os.mkdir(save_path)

    # print(out)
    if len(save_name) == 0 or save_name[0] != '-':
        save_name = "-" + save_name
    if k > 0:
        save_name += "-k=5"
    if ivf:
        save_name += "-ivf"
    if not_same_index:
        save_name += "-no-same"
    np.save(os.path.join(save_path, f"indices{save_name}.npy"), ids)
    np.save(os.path.join(save_path, f"scores{save_name}.npy"), scores)
    


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
        config["index_path"] = os.path.join(
            os.path.dirname(config["index_path"]),
            args.name,
            os.path.basename(config["index_path"]))
        config["index_source"] = os.path.join(
            os.path.dirname(config["index_source"]),
            args.name,
            os.path.basename(config["index_source"]))
        config["save_path"] = os.path.join(
            config["save_path"],
            args.name)
    # print(">>>> ", os.path.isdir(config["save_path"]), config["save_path"])
    if not os.path.isdir(config["save_path"]):
        # if not os.path.isdir(os.path.dirname(config["save_path"])):
        #     os.makedirs(os.path.dirname(config["save_path"]))
        os.makedirs(config["save_path"])
    # print(">>>> ", os.path.isdir(config["save_path"]))
            
    return retrieverCli(**config)

if __name__ == "__main__":
    main()