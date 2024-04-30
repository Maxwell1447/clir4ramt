import os
import torch
import numpy as np
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import PrependTokenDataset
from fairseq.data import iterators, data_utils
from fairseq.data import Dictionary, FairseqDataset
from torch.utils.data import DataLoader, Dataset
from functools import partial
from typing import *

def load_data_iter_from_path(data_path, split, src, tgt, max_positions=1024, max_tokens=2048, max_sentences=100):
    dataset, src_dict = load_data_from_path(data_path, split, src, tgt)
    return load_epoch_iter(
        dataset,
        seed=0,
        max_positions=max_positions,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=1,
        ignore_invalid_inputs=True
    ), src_dict

def load_data_from_path(data_path, split, src, tgt):
    src_dict = Dictionary.load(os.path.join(data_path, f"dict.{src}.txt"))
    tgt_dict = src_dict
    return load_data(data_path, split, src_dict, tgt_dict, src, tgt), src_dict


def load_data(
    data_path,
    split,
    src_dict,
    tgt_dict,
    src,
    tgt
):
    return load_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        True,
        "mmap",
        1,
        False,
        False,
        1024,
        1024,
        prepend_bos=True
    )

def load_epoch_iter(
    dataset,
    seed=0,
    max_positions=1024,
    max_tokens=3000,
    max_sentences=100,
    required_batch_size_multiple=2,
    ignore_invalid_inputs=True
):
    # get indices ordered by example size
    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()

    # filter examples that are too large
    if max_positions is not None:
        indices, _ = dataset.filter_indices_by_size(indices, max_positions)

    # create mini-batches with given size constraints
    batch_sampler = dataset.batch_by_size(
        indices,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
    )

    # return a reusable, sharded iterator
    epoch_iter = iterators.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_sampler,
        seed=seed,
        num_shards=1,
        shard_id=0,
        num_workers=2,
        epoch=1,
        buffer_size=0,
        return_dataloader=True
    )

    return epoch_iter

# class IndexerDataset(Dataset):
class IndexerDataset(FairseqDataset):
    def __init__(self, dataset, max_length=512, eos=2, pad=1):
        self.dataset = dataset
        self.eos = eos
        self.pad = pad
        self.max_length = max_length
        self.sizes = np.array([min(len(self.dataset[idx]), max_length) for idx in range(len(self))])
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if len(self.dataset[idx]) > self.max_length:
            x = self.dataset[idx]
            x = x[:self.max_length]
            x[-1] = self.eos
            return dict(id=torch.tensor(idx), tokens=x)
        return dict(id=torch.tensor(idx), tokens=self.dataset[idx])

    def ordered_indices(self):
        return np.argsort(self.sizes, kind="mergesort")

    def filter_indices_by_size(self, indices, max_position):
        return indices, None

    def num_tokens(self, index):
        return self.sizes[index]

    def num_tokens_vec(self, indices):
        return self.sizes[indices]

    def size(self, index):
        return self.sizes[index]

    def collater(self, samples, **kwargs):
        out = dict()
        assert len(samples) > 0
        for k in samples[0]:
            if samples[0][k].dim() == 0:
                out[k] = torch.tensor([dic[k] for dic in samples])
            else:
                out[k] = data_utils.collate_tokens([dic[k] for dic in samples], self.pad, **kwargs)
        return out

def collater(pad_idx, dict_list, **kwargs):
    out = dict()
    assert len(dict_list) > 0
    for k in dict_list[0]:
        if dict_list[0][k].dim() == 0:
            out[k] = torch.tensor([dic[k] for dic in dict_list])
        else:
            out[k] = data_utils.collate_tokens([dic[k] for dic in dict_list], pad_idx, **kwargs)
    return out

def load_monolingual_corpus(data_path=None, dict_path=None, batch_size=None):
    mono_dict = Dictionary.load(dict_path)
    dataset = data_utils.load_indexed_dataset(
        data_path,
        mono_dict,
        "mmap",
        combine=True,
    )
    dataset = IndexerDataset(PrependTokenDataset(dataset, mono_dict.bos()), max_length=512, eos=mono_dict.eos())
    return load_epoch_iter(
        dataset,
        seed=0,
        max_positions=1024,
        max_tokens=batch_size,
        max_sentences=100,
        required_batch_size_multiple=1,
        ignore_invalid_inputs=True
    ), mono_dict
    # return DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     collate_fn=partial(collater, mono_dict.pad_index)
    # ), mono_dict