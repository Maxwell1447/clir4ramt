"""Loss functions, optimizers, and schedulers."""
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


class LinearLRWithWarmup(LambdaLR):
    """
    Linear learning rate scheduler with linear warmup.
    Adapted from https://github.com/huggingface/transformers/blob/v4.23.0/src/transformers/optimization.py#L75
    
    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to LambdaLR
    warmup_steps: int
    total_steps: int
    """
    def __init__(self, *args, warmup_steps, total_steps, **kwargs):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(*args, **kwargs, lr_lambda=self.lr_lambda)
            
    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return max(
            0.0, 
            float(self.total_steps - current_step) / float(max(1, self.total_steps - self.warmup_steps))
        )

class InverseSqrtLRWithWarmup(LambdaLR):
    """
    Linear learning rate scheduler with linear warmup.
    Adapted from https://github.com/huggingface/transformers/blob/v4.23.0/src/transformers/optimization.py#L75
    
    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to LambdaLR
    warmup_steps: int
    total_steps: int
    """
    def __init__(self, *args, warmup_steps, total_steps, update_factor, **kwargs):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.update_factor = update_factor if update_factor is not None else (warmup_steps if warmup_steps > 10 else total_steps / 20)
        super().__init__(*args, **kwargs, lr_lambda=self.lr_lambda)
            
    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        # return 1 / np.sqrt(float(current_step - self.warmup_steps) / float(self.total_steps - self.warmup_steps) * self.update_factor ** 2 + 1)
        return 1 / np.sqrt(float(current_step - self.warmup_steps) / float(self.update_factor) + 1)

class LabelSmoothingLoss(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, logprobs, target):
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class MultiNLLLoss(nn.Module):
    """
    - sum_w log(p_w | X) 
    """

    def __init__(self, label_smoothing=0.0):
        super(MultiNLLLoss, self).__init__()
        self.ls = label_smoothing

    def forward(self, logprobs, target_mask):
        # logprobs : B x V
        # target_mask : B x V
        if self.ls > 0:
            smooth_loss = -logprobs.mean(-1).mean()
        else:
            smooth_loss = 0
        masked_logprobs = logprobs.masked_fill(~target_mask, 0)
        loss = -(masked_logprobs.sum(-1) / target_mask.sum(-1)).mean()
        return self.ls * smooth_loss + (1.0 - self.ls) * loss
        # else:
        #     if self.reduction == "mean":
        #         return -logprobs[target_mask].mean()
        #     else:
        #         return -logprobs[target_mask].sum()

class MSELevenshteinLoss(nn.Module):
    def __init__(self, alpha=0.6):
        super(MSELevenshteinLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, similarities, levs):
        pseudo_lev = torch.clamp(similarities - self.alpha, min=0) / (1 - self.alpha)
        print("dtype", similarities.dtype, levs.dtype)
        return self.mse(similarities.view(-1), levs.view(-1))
