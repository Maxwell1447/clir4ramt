import os
import argparse
import numpy as np
from tqdm import tqdm
import pyonmttok
from levenshtein import levenshtein as lev


def getargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True, help="data directory")
    parser.add_argument("--sub-dir-names", required=True, help="data directory")
    parser.add_argument("--retrieval-dir", required=True, help="retrieval directory")
    parser.add_argument("--retrieval-type", default="train-full-k=5")
    parser.add_argument("--rank", type=int, default=4, help="max rank of the match to score")
    parser.add_argument("--lang", default="fr", help="language id extension")
    parser.add_argument("--split", default="train", help="split used as queries in retrieval")

    return parser.parse_args()


def precompute_lev(data_path, retrieval_path, output_path, k=None):
    ids = np.load(retrieval_path)
    if k is not None:
        ids = ids[:, :k]
    with open(data_path) as f:
        tgt_sents = f.readlines()
    levs = np.empty(ids.shape, dtype=np.float32)
    for r in range(args.rank):
        tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
        for i in tqdm(range(len(ids))):
            if i == ids[i][r]:
                levs[i, r] = 1.0
            else:
                tok_tgt = tokenizer.tokenize(tgt_sents[i].rstrip())[0]
                tok_ret = tokenizer.tokenize(tgt_sents[ids[i][r]].rstrip())[0]
                levs[i, r] = lev(tok_tgt, tok_ret)
    np.save(output_path, levs)

def precompute_lev_external(data_path, external_data_path, retrieval_path, output_path, k=None):
    ids = np.load(retrieval_path)
    if k is not None:
        ids = ids[:, :k]
    with open(data_path) as f:
        tgt_sents = f.readlines()
    with open(external_data_path) as f:
        ext_sents = f.readlines()
    levs = np.empty(ids.shape, dtype=np.float32)
    for r in range(args.rank):
        tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)
        for i in tqdm(range(len(ids))):
            if i == ids[i][r]:
                levs[i, r] = 1.0
            else:
                tok_tgt = tokenizer.tokenize(ext_sents[i].rstrip())[0]
                tok_ret = tokenizer.tokenize(tgt_sents[ids[i][r]].rstrip())[0]
                levs[i, r] = lev(tok_tgt, tok_ret)
    np.save(output_path, levs)


def concatenate_values(retrieved_path, filename, names, subname="indices", need_offset=True):
    xs = list()
    offset = 0
    for name in names:
        x = np.load(os.path.join(retrieved_path, name, f"{subname}-{filename}"))
        if need_offset:
            x += offset
        xs.append(x)
        offset += len(x)
    xs = np.concatenate(xs)
    np.save(os.path.join(retrieved_path, f"{subname}-cat-{filename}"), xs)


if __name__ == "__main__":
    args = getargs()

    concatenate_values(args.retrieval_dir, f"{args.retrieval_type}.npy", args.sub_dir_names.split(','), subname="indices", need_offset=True)

    # for name in args.sub_dir_names.split(','):
    #     retrieval_path = os.path.join(args.retrieval_dir, name, f"indices-{args.retrieval_type}.npy")
    #     output_path = os.path.join(args.retrieval_dir, name, f"lev-{args.retrieval_type}.npy")
    #     data_path = os.path.join(args.data_dir, f"{name}.train.{args.lang}")
    #     if args.split != "train":
    #         external_data_path = os.path.join(args.data_dir, f"{name}.{args.split}.{args.lang}")
    #         precompute_lev_external(data_path, external_data_path, retrieval_path, output_path, k=args.rank)
    #     else:
    #         precompute_lev(data_path, retrieval_path, output_path, k=args.rank)

    # concatenate_values(args.retrieval_dir, f"{args.retrieval_type}.npy", args.sub_dir_names.split(','), subname="lev", need_offset=False)
    # concatenate_values(args.retrieval_dir, f"{args.retrieval_type}.npy", args.sub_dir_names.split(','), subname="scores", need_offset=False)
