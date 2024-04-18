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

    def __init__(self, reduction="mean"):
        super(MultiNLLLoss, self).__init__()
        self.reduction = "mean"

    def forward(self, logprobs, target):
        # logprobs : B x V
        # target : (B x V)
        if self.reduction == "mean":
            return -logprobs[target].mean()
        else:
            return -logprobs[target].sum()

class BOWModule(nn.Module):
    """BOW loss to predict terms in the other language.
    """
    def __init__(self, d_hidden, voc_size, factor=0.1):
        super(BOWModule, self).__init__()
        self.voc_size = voc_size
        self.factor = factor
        self.num_spec = 4
        self.voc_linear = nn.Linear(d_hidden, voc_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_fct = MultiNLLLoss(reduction='mean')
        # self.loss_fct = nn.NLLLoss(reduction='mean')

    def forward(self, last_hidden_state_cls, other_sentence):
        # last_hidden_state_cls : B x d
        # other_sentence : B x L
        one_hot = torch.zeros(other_sentence.size(0), self.voc_size + self.num_spec, device=other_sentence.device, dtype=torch.bool)
        # tgt : (B x V)
        tgt = one_hot.scatter_(1, other_sentence, 1)[:, self.num_spec:].reshape(other_sentence.size(0), self.voc_size)
        ## out : (B x V) x 2
        # out : B x V
        out_ = self.voc_linear(last_hidden_state_cls) #.view(other_sentence.size(0) * self.voc_size)
        # out = torch.zeros((other_sentence.size(0) * self.voc_size, 2), dtype=out_.dtype, device=other_sentence.device)
        # out[:, 1] = out_
        out = self.log_softmax(out_)
        # return self.factor * self.loss_fct(out, tgt)
        return dict(loss=self.factor * self.loss_fct(out, tgt), logprobs=out, target=tgt)
        