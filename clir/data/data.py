import os
import torch
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import iterators, data_utils
from fairseq.data import Dictionary
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
    )

    return epoch_iter
