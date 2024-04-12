# coding: utf-8

from pathlib import Path
from docopt import docopt
import json
import os
import warnings

import re
import string

from datasets import load_dataset, Dataset, load_from_disk, set_caching_enabled
import transformers


def verbose_load_from_disk(dataset_path):
    print(f"Loading '{dataset_path}'")
    dataset = load_from_disk(dataset_path)
    print(dataset)
    return dataset

def get_pretrained(pretrained_model_name_or_path, **kwargs):
    if pretrained_model_name_or_path is None:
        model = transformers(Class.config_class(**kwargs))
        print(f"Randomly initialized model:\n{model}")
    else:
        model = transformers.from_pretrained(pretrained_model_name_or_path, **kwargs)        
    return model


def load_pretrained_in_kwargs(kwargs):
    """Recursively loads pre-trained models/tokenizer in kwargs using get_pretrained"""
    # base case: load pre-trained model
    if 'class_name' in kwargs:
        return get_pretrained(**kwargs)
    # recursively look in the child arguments
    for k, v in kwargs.items():
        if isinstance(v, dict):
            kwargs[k] = load_pretrained_in_kwargs(v)
        # else keep as is
    return kwargs


if __name__ == '__main__':
    args = docopt(__doc__)
    ...
