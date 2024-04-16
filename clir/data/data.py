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
        max_positions=1024,
        max_tokens=3000,
        max_sentences=100,
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

# from torch.utils.data import Dataset, DataLoader
# class DataLoaderEpoch(DataLoader):
#     r"""
#     Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.

#     The :class:`~torch.utils.data.DataLoader` supports both map-style and
#     iterable-style datasets with single- or multi-process loading, customizing
#     loading order and optional automatic batching (collation) and memory pinning.

#     See :py:mod:`torch.utils.data` documentation page for more details.

#     Args:
#         dataset (Dataset): dataset from which to load the data.
#         batch_size (int, optional): how many samples per batch to load
#             (default: ``1``).
#         shuffle (bool, optional): set to ``True`` to have the data reshuffled
#             at every epoch (default: ``False``).
#         sampler (Sampler or Iterable, optional): defines the strategy to draw
#             samples from the dataset. Can be any ``Iterable`` with ``__len__``
#             implemented. If specified, :attr:`shuffle` must not be specified.
#         batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
#             returns a batch of indices at a time. Mutually exclusive with
#             :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
#             and :attr:`drop_last`.
#         num_workers (int, optional): how many subprocesses to use for data
#             loading. ``0`` means that the data will be loaded in the main process.
#             (default: ``0``)
#         collate_fn (Callable, optional): merges a list of samples to form a
#             mini-batch of Tensor(s).  Used when using batched loading from a
#             map-style dataset.
#         pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
#             into device/CUDA pinned memory before returning them.  If your data elements
#             are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
#             see the example below.
#         drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
#             if the dataset size is not divisible by the batch size. If ``False`` and
#             the size of dataset is not divisible by the batch size, then the last batch
#             will be smaller. (default: ``False``)
#         timeout (numeric, optional): if positive, the timeout value for collecting a batch
#             from workers. Should always be non-negative. (default: ``0``)
#         worker_init_fn (Callable, optional): If not ``None``, this will be called on each
#             worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
#             input, after seeding and before data loading. (default: ``None``)
#         multiprocessing_context (str or multiprocessing.context.BaseContext, optional): If
#             ``None``, the default `multiprocessing context`_ of your operating system will
#             be used. (default: ``None``)
#         generator (torch.Generator, optional): If not ``None``, this RNG will be used
#             by RandomSampler to generate random indexes and multiprocessing to generate
#             ``base_seed`` for workers. (default: ``None``)
#         prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
#             in advance by each worker. ``2`` means there will be a total of
#             2 * num_workers batches prefetched across all workers. (default value depends
#             on the set value for num_workers. If value of num_workers=0 default is ``None``.
#             Otherwise, if value of ``num_workers > 0`` default is ``2``).
#         persistent_workers (bool, optional): If ``True``, the data loader will not shut down
#             the worker processes after a dataset has been consumed once. This allows to
#             maintain the workers `Dataset` instances alive. (default: ``False``)
#         pin_memory_device (str, optional): the device to :attr:`pin_memory` to if ``pin_memory`` is
#             ``True``.


#     .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
#                  cannot be an unpicklable object, e.g., a lambda function. See
#                  :ref:`multiprocessing-best-practices` on more details related
#                  to multiprocessing in PyTorch.

#     .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
#                  When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
#                  it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
#                  rounding depending on :attr:`drop_last`, regardless of multi-process loading
#                  configurations. This represents the best guess PyTorch can make because PyTorch
#                  trusts user :attr:`dataset` code in correctly handling multi-process
#                  loading to avoid duplicate data.

#                  However, if sharding results in multiple workers having incomplete last batches,
#                  this estimate can still be inaccurate, because (1) an otherwise complete batch can
#                  be broken into multiple ones and (2) more than one batch worth of samples can be
#                  dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
#                  cases in general.

#                  See `Dataset Types`_ for more details on these two types of datasets and how
#                  :class:`~torch.utils.data.IterableDataset` interacts with
#                  `Multi-process data loading`_.

#     .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
#                  :ref:`data-loading-randomness` notes for random seed related questions.

#     .. _multiprocessing context:
#         https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
#     """

#     num_workers: int
#     pin_memory: bool
#     _iterator : Optional['_BaseDataLoaderIter']
#     __initialized = False

#     def __init__(self, dataset,
#                  shuffle: Optional[bool] = None,
#                  pin_memory: bool = False,
#                  persistent_workers: bool = False):

#         self.dataset = dataset

#         self.shuffle = bool(shuffle)

#         self._iterator = None

#         torch.set_vital('Dataloader', 'enabled', 'True')  # type: ignore[attr-defined]

#     def _get_iterator(self) -> '_BaseDataLoaderIter':
#         return self.dataset.next_epoch_itr(shuffle=self.shuffle)

#     @property
#     def multiprocessing_context(self):
#         return self.__multiprocessing_context

#     @multiprocessing_context.setter
#     def multiprocessing_context(self, multiprocessing_context):
#         if multiprocessing_context is not None:
#             if self.num_workers > 0:
#                 if isinstance(multiprocessing_context, str):
#                     valid_start_methods = multiprocessing.get_all_start_methods()
#                     if multiprocessing_context not in valid_start_methods:
#                         raise ValueError(
#                             'multiprocessing_context option '
#                             f'should specify a valid start method in {valid_start_methods!r}, but got '
#                             f'multiprocessing_context={multiprocessing_context!r}')
#                     multiprocessing_context = multiprocessing.get_context(multiprocessing_context)

#                 if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
#                     raise TypeError('multiprocessing_context option should be a valid context '
#                                     'object or a string specifying the start method, but got '
#                                     f'multiprocessing_context={multiprocessing_context}')
#             else:
#                 raise ValueError('multiprocessing_context can only be used with '
#                                  'multi-process loading (num_workers > 0), but got '
#                                  f'num_workers={self.num_workers}')

#         self.__multiprocessing_context = multiprocessing_context

#     # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
#     # since '_BaseDataLoaderIter' references 'DataLoader'.
#     def __iter__(self) -> '_BaseDataLoaderIter':
#         # When using a single worker the returned iterator should be
#         # created everytime to avoid resetting its state
#         # However, in the case of a multiple workers iterator
#         # the iterator is only created once in the lifetime of the
#         # DataLoader object so that workers can be reused
#         return self._get_iterator()

#     # @property
#     # def _index_sampler(self):
#     #     # The actual sampler used for generating indices for `_DatasetFetcher`
#     #     # (see _utils/fetch.py) to read data at each time. This would be
#     #     # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
#     #     # We can't change `.sampler` and `.batch_sampler` attributes for BC
#     #     # reasons.
#     #     if self._auto_collation:
#     #         return self.batch_sampler
#     #     else:
#     #         return self.sampler

#     def __len__(self) -> int:
#         return 

#     # def check_worker_number_rationality(self):
#     #     # This function check whether the dataloader's worker number is rational based on
#     #     # current system's resource. Current rule is that if the number of workers this
#     #     # Dataloader will create is bigger than the number of logical cpus that is allowed to
#     #     # use, than we will pop up a warning to let user pay attention.
#     #     #
#     #     # eg. If current system has 2 physical CPUs with 16 cores each. And each core support 2
#     #     #     threads, then the total logical cpus here is 2 * 16 * 2 = 64. Let's say current
#     #     #     DataLoader process can use half of them which is 32, then the rational max number of
#     #     #     worker that initiated from this process is 32.
#     #     #     Now, let's say the created DataLoader has num_works = 40, which is bigger than 32.
#     #     #     So the warning message is triggered to notify the user to lower the worker number if
#     #     #     necessary.
#     #     #
#     #     #
#     #     # [Note] Please note that this function repects `cpuset` only when os.sched_getaffinity is
#     #     #        available (available in most of Linux system, but not OSX and Windows).
#     #     #        When os.sched_getaffinity is not available, os.cpu_count() is called instead, but
#     #     #        it doesn't repect cpuset.
#     #     #        We don't take threading into account since each worker process is single threaded
#     #     #        at this time.
#     #     #
#     #     #        We don't set any threading flags (eg. OMP_NUM_THREADS, MKL_NUM_THREADS, etc)
#     #     #        other than `torch.set_num_threads` to 1 in the worker process, if the passing
#     #     #        in functions use 3rd party modules that rely on those threading flags to determine
#     #     #        how many thread to create (eg. numpy, etc), then it is caller's responsibility to
#     #     #        set those flags correctly.
#     #     def _create_warning_msg(num_worker_suggest, num_worker_created, cpuset_checked):

#     #         suggested_max_worker_msg = ((
#     #             "Our suggested max number of worker in current system is {}{}, which is smaller "
#     #             "than what this DataLoader is going to create.").format(
#     #                 num_worker_suggest,
#     #                 ("" if cpuset_checked else " (`cpuset` is not taken into account)"))
#     #         ) if num_worker_suggest is not None else (
#     #             "DataLoader is not able to compute a suggested max number of worker in current system.")

#     #         warn_msg = (
#     #             "This DataLoader will create {} worker processes in total. {} "
#     #             "Please be aware that excessive worker creation might get DataLoader running slow or even freeze, "
#     #             "lower the worker number to avoid potential slowness/freeze if necessary.").format(
#     #                 num_worker_created,
#     #                 suggested_max_worker_msg)
#     #         return warn_msg

#     #     if not self.num_workers or self.num_workers == 0:
#     #         return

#     #     # try to compute a suggested max number of worker based on system's resource
#     #     max_num_worker_suggest = None
#     #     cpuset_checked = False
#     #     if hasattr(os, 'sched_getaffinity'):
#     #         try:
#     #             max_num_worker_suggest = len(os.sched_getaffinity(0))
#     #             cpuset_checked = True
#     #         except Exception:
#     #             pass
#     #     if max_num_worker_suggest is None:
#     #         # os.cpu_count() could return Optional[int]
#     #         # get cpu count first and check None in order to satisfy mypy check
#     #         cpu_count = os.cpu_count()
#     #         if cpu_count is not None:
#     #             max_num_worker_suggest = cpu_count

#     #     if max_num_worker_suggest is None:
#     #         warnings.warn(_create_warning_msg(
#     #             max_num_worker_suggest,
#     #             self.num_workers,
#     #             cpuset_checked))
#     #         return

#     #     if self.num_workers > max_num_worker_suggest:
#     #         warnings.warn(_create_warning_msg(
#     #             max_num_worker_suggest,
#     #             self.num_workers,
#     #             cpuset_checked))

# class RawDataset(torch.utils.Dataset):
#     def __init__(self, raw_filename, src, tgt):
#         self.src_filename = f"{raw_filename}.{src}"
#         self.tgt_filename = f"{raw_filename}.{tgt}"
