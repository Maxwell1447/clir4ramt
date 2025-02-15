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
    full_state_update: bool = True
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
    full_state_update: bool = True
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
    full_state_update: bool = True
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

class InBatchAccuracyContrastive(Metric):
    full_state_update: bool = True
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("hits_at_rank", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, similarities: torch.Tensor, levs: torch.Tensor):
        bsz, k = similarities.shape
        self.hits_at_rank += (
            similarities.argsort(-1, descending=True) == 
            levs.argsort(-1, descending=True)
        ).sum()
        self.total += bsz * k

    def compute(self):
        return self.hits_at_rank / self.total

class InBatchNDCG(Metric):
    full_state_update: bool = True
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("ndcg", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, similarities: torch.Tensor, levs: torch.Tensor):
        bsz, k = similarities.shape
        ## normalize
        similarities = (similarities + 1) / 2
        true_ranking = levs.argsort(-1, descending=True)
        norm = similarities.sort(-1, descending=True)[0]
        ndcg = (
            (
                similarities.gather(1, true_ranking) / 
                torch.log2(2 + torch.arange(k, dtype=similarities.dtype, device=similarities.device)).view(1, -1)
            ).sum(-1) / 
            (
                norm / 
                torch.log2(2 + torch.arange(k, dtype=similarities.dtype, device=similarities.device)).view(1, -1)
            ).sum(-1)
        ).sum()
        assert (ndcg / bsz).item() <= 1 and (ndcg / bsz).item() >= 0, f"{similarities.gather(1, true_ranking)}\n vs \n {norm}\n log = {torch.log2(2 + torch.arange(k, dtype=similarities.dtype, device=similarities.device)).view(1, -1)} \n bsz = {bsz}"
        self.ndcg += ndcg
        self.total += bsz

    def compute(self):
        return self.ndcg / self.total
