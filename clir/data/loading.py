# coding: utf-8

from pathlib import Path
import json
import os
import warnings

import re
import string

import transformers
# from clir.models import BertWithCustomEmbedding
from transformers import BertModel, XLMModel, XLMRobertaModel

ClassModel = BertModel


def get_pretrained(pretrained_model_name_or_path, **kwargs):
    if pretrained_model_name_or_path is None:
        # print("kwargs", kwargs)
        # print(ClassModel.config_class())
        # print(ClassModel.config_class(**kwargs))
        config = ClassModel.config_class(**kwargs)
        model = ClassModel(config)
        print(f"Randomly initialized model:\n{model}")
    else:
        model = transformers.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config = None
    return model, config


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
    ...
