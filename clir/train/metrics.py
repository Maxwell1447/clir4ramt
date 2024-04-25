"""Metrics to be used in trainer."""

import torch
from torchmetrics import Metric, MetricCollection


def get_ranking(log_probs):
    """
    Compute rank of the diagonal
    """
    bsz, _ = log_probs.shape
    sorted_index = log_probs.argsort(axis=1, descending=True)
    rankings = torch.empty_like(sorted_index)
    rankings.scatter_(1, sorted_index, torch.arange(bsz, device=sorted_index.device).view(1, bsz).expand_as(sorted_index))
    diag = torch.diagonal(rankings)
    return diag

class InBatchAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("hits_at_1", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, log_probs: torch.Tensor):
        bsz, _ = log_probs.shape
        self.hits_at_1 += (get_ranking(log_probs) == 0).sum()
        self.total += bsz

    def compute(self):
        return self.hits_at_1 / self.total

class InBatchMRR(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("mrr", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, log_probs: torch.Tensor):
        bsz, _ = log_probs.shape
        self.mrr += (1 / (get_ranking(log_probs) + 1)).sum()
        self.total += bsz

    def compute(self):
        return self.mrr / self.total

class BOWRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("hits", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, log_probs: torch.Tensor, target: torch.Tensor):
        bsz, _ = log_probs.shape
        # log_probs.argsort(1)
        self.hits += torch.gather(target, 1, log_probs.argsort(1, descending=True))[target.long().sort(1, descending=True)[0].bool()].sum()
        self.total += target.sum()

    def compute(self):
        return self.hits / self.total