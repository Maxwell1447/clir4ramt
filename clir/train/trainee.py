"""
Trainee is a pl.LightningModule that computes the loss so it is compatible with Trainer.
"""
import warnings
from functools import partial
import re
from pathlib import Path
import json
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
from transformers import BertModel
# from clir.models import BertWithCustomEmbedding

from ..data.loading import get_pretrained
from .optim import LinearLRWithWarmup


class Trainee(pl.LightningModule):
    """
    Base class for all Trainee models (to be trained by a Trainer)
    
    Parameters
    ----------    
    *args, **kwargs: additionnal arguments are passed to pl.LightningModule
    freeze_regex: str, optional
        represents a regex used to match the model parameters to freeze
        (i.e. set ``requires_grad = False``).
        Defaults to None (keep model fully-trainable)
    gradient_checkpointing: bool, optional
    lr, eps, weight_decay: float, optional
    betas: Tuple[float], optional    
    warmup_steps: int, optional
        Defaults to no warm-up
    output_cpu: bool, optional
    """
    def __init__(self, *args, freeze_regex=None, gradient_checkpointing=False,
                 warmup_steps=0, lr=2e-5, betas=(0.9, 0.999), eps=1e-08, 
                 weight_decay=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_regex = freeze_regex
        self.gradient_checkpointing = gradient_checkpointing
        # self.weights_to_log = {}
        
        # scheduling and optimization
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay        
        self.param_groups = self.parameters()
        
    # should be called at the end of each subclass __init__
    def post_init(self):
        if self.freeze_regex is not None:
            self.freeze(self.freeze_regex)        
        if self.gradient_checkpointing:
            self.gradient_checkpointing_enable()
        
    def step(self, batch, batch_idx):
        raise NotImplementedError("Subclass and implement step.")
    
    def eval_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)
    
    def log(self, name, value, **kwargs):
        """Ignores None values."""
        if value is None:
            return None
        super().log(name, value, **kwargs)
            
    def training_step(self, batch, batch_idx):
        """Step and log training metrics"""
        outputs = self.step(batch, batch_idx)
        self.log("train/loss", outputs['loss'])
        # for name, tensor in self.weights_to_log.items():
        #     self.log(f"weights/{name}", tensor.cpu().detach().item())

        return outputs
    
    def validation_step(self, batch, batch_idx):
        """Step and log validation metrics"""
        outputs = self.eval_step(batch, batch_idx)
        self.log("eval/loss", outputs['loss'])
        return outputs
    
    def test_step(self, batch, batch_idx):
        """Step and log test metrics"""
        outputs = self.eval_step(batch, batch_idx)
        self.log("test/loss", outputs['loss'])
        return outputs
    
    def eval_epoch_end(self, eval_outputs):
        warnings.warn("eval_epoch_end is not implemented.")
        return {}
    
    def on_validation_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        loss = self.eval_epoch_end(*args, **kwargs)['mean-loss']
        self.log(f"eval/loss", loss)
            
    def on_test_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        loss = self.eval_epoch_end(*args, **kwargs)['mean-loss']
        log_dir = Path(self.trainer.log_dir)
        self.log(f"test/loss", loss)
                    
    def freeze(self, regex):
        """
        Overrides freeze to freeze only parameters that match the regex.
        Caveat: does not call .eval() so does not disable Dropout
        """
        regex = re.compile(regex)
        total, frozen = 0, 0
        print("Model parameters:\n"+"Name".ljust(120)+"\t#Trainable\t#Total")
        for name, param in self.named_parameters():
            numel = param.numel()
            if regex.match(name):
                param.requires_grad = False
                frozen += numel
            total += numel
            print(f"{name.ljust(120)}\t{(numel if param.requires_grad else 0):,d}\t{numel:,d}")
        print(f"Froze {frozen:,d} parameters out of {total:,d}")
        
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.param_groups, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        
        # FIXME: this will be overwritten when loading state from ckpt_path
        # so if you want to keep training by increasing total_steps, 
        # your LR will be 0 if the ckpt reached the previously set total_steps
        total_steps=self.trainer.estimated_stepping_batches
        scheduler = LinearLRWithWarmup(
            optimizer,
            warmup_steps=self.warmup_steps, total_steps=total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
        
    
    #####################################################
    # gradient checkpointing: adapted from transformers #
    #####################################################
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
            
    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(getattr(m, "gradient_checkpointing", False) for m in self.modules())

        
# def _get_bert(dpr_encoder):
#     if hasattr(dpr_encoder, 'question_encoder'):
#         return dpr_encoder.question_encoder.bert_model
#     return dpr_encoder.ctx_encoder.bert_model


class BiEncoder(Trainee):
    """    
    The training objective is to minimize the negative log-likelihood of the similarities (dot product)
    between the questions and the passages embeddings, as described in [3]_.
    """
    def __init__(
        self,
        *args,
        model_name_or_path=None,
        vocab_size=30145,
        pad_token_id=0,
        type_vocab_size=2,
        **kwargs):
        super().__init__(*args, **kwargs)        
        # default to symmetric encoders
        # init encoders
        self.pad_token_id = pad_token_id
        self.src_model = get_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            type_vocab_size=type_vocab_size,
        )
        # self.src_model = get_pretrained(BertModel, pretrained_model_name_or_path=model_name_or_path)
        self.tgt_model = self.src_model
        
        # loss and metrics
        self.log_softmax = nn.LogSoftmax(1)
        self.loss_fct = nn.NLLLoss(reduction='mean')
        
        self.post_init()

    def make_input_from_fairseq(self, input):
        return dict(
            src=dict(
                input_ids=input["net_input"]["src_tokens"],
                attention_mask=input["net_input"]["src_tokens"].ne(self.pad_token_id)
            ),
            tgt=dict(
                input_ids=input["target"],
                attention_mask=input["target"].ne(self.pad_token_id)
            )
        )
        
    def forward(self, input):
        """        
        Parameters
        ----------
        input: dict
        """
        # print("input\n", input)
        if "net_input" in input:
            input = self.make_input_from_fairseq(input)
        # embed questions and contexts
        src_out = self.src_model(**input["src"])
        tgt_out = self.tgt_model(**input["tgt"])
        # print(src_out)
        # print(src_out.last_hidden_state[:, 0, :])

        return dict(src=src_out.last_hidden_state[:, 0, :], tgt=tgt_out.last_hidden_state[:, 0, :])

    def step(self, inputs, _):
        """
        Calculates In-batch negatives schema loss and supports to run it in DDP mode 
        by exchanging the representations across all the nodes.
        
        Adapted from https://github.com/facebookresearch/DPR/blob/main/train_dense_encoder.py
        and https://github.com/Lightning-AI/lightning/discussions/14390
        
        Notes
        -----
        This means that the whole representations of questions and contexts, and their similarity matrix, must fit on a single GPU.
        """            
        outputs = self(inputs)
        
        ##### FOR MULTIPROCESSING sync
        src_out = self.all_gather(outputs["src"], sync_grads=True)
        tgt_out = self.all_gather(outputs["tgt"], sync_grads=True)
        
        # reshape after all_gather
        if src_out.ndim > 2:
            n_gpus, N, _ = src_out.shape
            src_out = src_out.view(n_gpus*N, -1)
            tgt_out = tgt_out.view(n_gpus*N, -1)

        # compute similarity
        similarities = src_out @ tgt_out.T  # (B, B)
        log_probs = self.log_softmax(similarities)

        loss = self.loss_fct(log_probs, torch.arange(len(similarities), device=similarities.device))
        return dict(loss=loss, log_probs=log_probs)
                    
    def eval_step(self, inputs, batch_idx):
        model_outputs = self.step(inputs, batch_idx)
        # metrics = batch_retrieval(model_outputs['log_probs'])
        return dict(loss=model_outputs['loss'])
                
    def eval_epoch_end(self, eval_outputs):
        loss = torch.mean([output['loss'] for output in eval_outputs])
        return {'mean-loss': loss}
    
    def save_pretrained(self, ckpt_path):
        self.src_model.save_pretrained(ckpt_path)
