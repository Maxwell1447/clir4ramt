"""Metrics to be used in trainer."""

import torch
# import warnings
# from collections import Counter


# def accumulate_batch_metrics(batch_metrics):    
#     metrics = Counter()   
#     for metric in batch_metrics:
#         for k, v in metric.items():
#             metrics[k] += v
#     effective_size = metrics.pop("batch_size") - metrics.pop("ignored_predictions", 0)
#     for k, v in metrics.items():
#         metrics[k] = v/effective_size
#     return metrics


def batch_metrics(log_probs):
    bsz, _ = log_probs.shape
    # use argsort to rank the passages w.r.t. their log-probability
    rankings = log_probs.argsort(axis=1, descending=True)
    diag = torch.diagonal(rankings, 0)
    hits_at_1 = (diag == 0).float().mean().item()
    mrr = (1 / (diag + 1)).float().mean().item()

    # for ranking, label in zip(rankings, range(bsz)):
    #     if ranking[0] == label:
    #         hits_at_1 += 1
    #     # +1 to count from 1 instead of 0
    #     rank = (ranking == label).nonzero()[0].item() + 1
    #     mrr += 1/rank    
    return {"MRR@N": mrr, "hits@1": hits_at_1, "bsz": bsz}


# def retrieval(eval_outputs, ignore_index=-100, output_key='log_probs'):
#     """
#     Computes metric for retrieval training (at the batch-level)
    
#     Parameters
#     ----------
#     eval_outputs: List[dict[str, Tensor]]
#         Contains log_probs and labels for all batches in the evaluation step (either validation or test)
#     ignore_index: int, optional
#         Labels with this value are not taken into account when computing metrics.
#         Defaults to -100
#     output_key: str, optional
#         Name of the model output in eval_outputs
#     """
#     metrics = {}    
#     mrr, hits_at_1, ignored_predictions, dataset_size = 0, 0, 0, 0
#     for batch in eval_outputs:
#         log_probs = batch[output_key].numpy()
#         labels = batch['labels'].numpy()
#         batch_size, _ = log_probs.shape
#         dataset_size += batch_size
#         # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)
#         rankings = (-log_probs).argsort(axis=1)
#         for ranking, label in zip(rankings, labels):
#             if label == ignore_index:
#                 ignored_predictions += 1
#                 continue
#             if ranking[0] == label:
#                 hits_at_1 += 1
#             # +1 to count from 1 instead of 0
#             rank = (ranking == label).nonzero()[0].item() + 1
#             mrr += 1/rank    
#     metrics["MRR@N*M"] = mrr / (dataset_size-ignored_predictions)
#     metrics["hits@1"] = hits_at_1 / (dataset_size-ignored_predictions)
#     return metrics
