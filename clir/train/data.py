import warnings
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from clir.data.data import load_data_from_path, load_data_iter_from_path

# class DataModuleRaw(pl.LightningDataModule):
#     """
#     Parameters
#     ----------
#     tokenizer_class: str
#     tokenizer_name_or_path: str
#     dataset_path: str
#     batch_size: int, optional
#     tokenization_kwargs: dict, optional
#     loader_kwargs: dict, optional
#     """
#     def __init__(self, tokenizer_class, tokenizer_name_or_path, 
#                  dataset_path,
#                  batch_size=8, tokenization_kwargs=None, loader_kwargs={}):
#         self.tokenizer = ... # get_pretrained(tokenizer_class, pretrained_model_name_or_path=tokenizer_name_or_path)
#         self.dataset_path = dataset_path
#         self.batch_size = batch_size
#         default_tokenization_kwargs = dict(
#             return_tensors='pt', 
#             padding='longest', 
#             truncation=True, 
#             return_overflowing_tokens=False
#         )
#         if tokenization_kwargs is not None:
#             default_tokenization_kwargs.update(tokenization_kwargs)
#         self.tokenization_kwargs = default_tokenization_kwargs
#         self.loader_kwargs = loader_kwargs

#     def setup(self, stage=None):
#         if self.dataset_path is None:
#             self.dataset = DatasetDict()
#             for subset in ['train', 'validation', 'test']:
#                 subset_path = getattr(self, subset+'_path', None)
#                 if subset_path is not None:
#                     self.dataset[subset] = verbose_load_from_disk(subset_path)
#         else:
#             self.dataset = verbose_load_from_disk(self.dataset_path)
#         if self.dataset_format is not None:
#             self.dataset.set_format(**self.dataset_format)
#         elif self.keep_dataset_columns is not None:
#             for name, subset in self.dataset.items():
#                 self.dataset[name] = keep_columns(subset, self.keep_dataset_columns)

class DataModuleMMap(pl.LightningDataModule):
    """
    Parameters
    ----------
    dict_path: str
    dataset_path: str
    src_lang: str
    tgt_lang: str
    train: bool, optional
    valid: bool, optional
    test: bool, optional
    max_tokens: int, optional
    max_position: int, optional
    max_sentences: int, optional
    """
    def __init__(self, dict_path, dataset_path, src_lang, tgt_lang, train=True, valid=True, test=False, max_positions=1024, max_tokens=2048, max_sentences=100):
        super().__init__()
        if train:
            self.train, self.dict = load_data_iter_from_path(dataset_path, "train", src_lang, tgt_lang, max_positions=max_positions, max_tokens=max_tokens, max_sentences=max_sentences)
        else:
            self.train = None
        if valid:
            self.valid, self.dict = load_data_iter_from_path(dataset_path, "valid", src_lang, tgt_lang, max_positions=max_positions, max_tokens=max_tokens, max_sentences=max_sentences)
        else:
            self.valid = None
        if test:
            self.test, self.dict = load_data_iter_from_path(dataset_path, "test", src_lang, tgt_lang, max_positions=max_positions, max_tokens=max_tokens, max_sentences=max_sentences)
        else:
            self.test = None
        
    # def setup(self, *args, **kwargs):
    #     ...
    
    # def prepare_data(self):
    #     ...

    def train_dataloader(self):    
        if self.train is not None:
            return self.train.next_epoch_itr(shuffle=True)
        return None

    def val_dataloader(self):
        if self.valid is not None:
            return self.valid.next_epoch_itr(shuffle=False)
        return None

    def test_dataloader(self):
        if self.test is not None:
            return self.test.next_epoch_itr(shuffle=False)
        return None


# class DataModule(pl.LightningDataModule):
#     """
#     Base class for all data modules. 
#     It has a tokenizer and handles dataset loading with train/validation/test subsets.
#     For multimodal models, it can also handle image features or pixels using ImageFormatter

#     Parameters
#     ----------
#     tokenizer_class: str
#         Name of a transformers.PreTrainedTokenizer subclass
#     tokenizer_name_or_path: str
#         see transformers.PreTrainedTokenizer.from_pretrained
#     dataset_path: str, optional
#         Path to a DatasetDict that should have 'train', 'validation' and 'test' subsets.
#         Alternatively, you can specify those using the dedicated variables.
#     train_path, validation_path, test_path: str, optional
#     batch_size, train_batch_size, eval_batch_size: int, optional
#         batch_size is needed to be able to tune it automatically using auto_scale_batch_size in Trainer
#         It is overriden by train_batch_size, eval_batch_size 
#         (if you want to use different batch sizes for training and evaluation)    
#     M: int, optional
#         Number of passages (relevant or irrelevant) per question in a batch
#         Defaults to 24
#     n_relevant_passages: int, optional
#         Defaults to 1
#     keep_dataset_columns: list, optional
#         Keep only these features in the dataset (defaults to keep everything)
#     tokenization_kwargs: dict, optional
#         To be passed to self.tokenizer
#     image_kwargs: dict, optional
#         Passed to ImageFormatter. Optional for text-only models.
#     loader_kwargs: dict, optional
#         Passed to the data loaders (e.g. self.train_dataloader())
#     dataset_format: dict, optional
#         see Dataset.set_format
#         Overrides keep_dataset_columns.
#     input_key: str, optional
#         Holds input text (e.g. question, caption), defaults to 'input'
#     """
#     def __init__(self, tokenizer_class, tokenizer_name_or_path, 
#                  dataset_path=None, train_path=None, validation_path=None, test_path=None, 
#                  batch_size=8, train_batch_size=None, eval_batch_size=None, 
#                  M=24, n_relevant_passages=1, keep_dataset_columns=None,
#                  tokenization_kwargs=None, image_kwargs={}, loader_kwargs={}, 
#                  dataset_format=None, input_key='input'):
#         super().__init__()
#         self.tokenizer = get_pretrained(tokenizer_class, pretrained_model_name_or_path=tokenizer_name_or_path)
#         self.dataset_path = dataset_path
#         self.train_path = train_path        
#         self.validation_path = validation_path
#         self.test_path = test_path
#         self.batch_size = batch_size
#         self.train_batch_size = train_batch_size
#         self.eval_batch_size = eval_batch_size
#         self.M = M
#         self.n_relevant_passages = n_relevant_passages
#         self.keep_dataset_columns = set(keep_dataset_columns) if keep_dataset_columns is not None else None
#         self.dataset_format = dataset_format
#         self.input_key = input_key
        
#         # useful in some corner-cases like ICT. False for every other data modules
#         self.shuffle_eval = False
#         default_tokenization_kwargs = dict(
#             return_tensors='pt', 
#             padding='longest', 
#             truncation=True, 
#             return_overflowing_tokens=False
#         )
#         if tokenization_kwargs is not None:
#             default_tokenization_kwargs.update(tokenization_kwargs)
#         self.tokenization_kwargs = default_tokenization_kwargs
#         self.image_formatter = ImageFormatter(**image_kwargs)
#         self.loader_kwargs = loader_kwargs
        
#     def setup(self, stage=None):
#         if self.dataset_path is None:
#             self.dataset = DatasetDict()
#             for subset in ['train', 'validation', 'test']:
#                 subset_path = getattr(self, subset+'_path', None)
#                 if subset_path is not None:
#                     self.dataset[subset] = verbose_load_from_disk(subset_path)
#         else:
#             self.dataset = verbose_load_from_disk(self.dataset_path)
#         if self.dataset_format is not None:
#             self.dataset.set_format(**self.dataset_format)
#         elif self.keep_dataset_columns is not None:
#             for name, subset in self.dataset.items():
#                 self.dataset[name] = keep_columns(subset, self.keep_dataset_columns)
            
#     def train_dataloader(self):    
#         if 'train' not in self.dataset:
#             return None    
#         # set here and not in __init__ so that it will be reset properly in Trainer.reset_train_dataloader,
#         # which is called during auto_scale_batch_size
#         batch_size = self.train_batch_size if self.train_batch_size is not None else self.batch_size
#         return DataLoader(
#             self.dataset['train'], 
#             batch_size=batch_size,
#             collate_fn=self.collate_fn,
#             shuffle=True,
#             **self.loader_kwargs
#         )

#     def val_dataloader(self):
#         if 'validation' not in self.dataset:
#             return None
#         batch_size = self.eval_batch_size if self.eval_batch_size is not None else self.batch_size
#         return DataLoader(
#             self.dataset['validation'], 
#             batch_size=batch_size, 
#             collate_fn=self.collate_fn,
#             shuffle=self.shuffle_eval,
#             **self.loader_kwargs
#         )

#     def test_dataloader(self):
#         if 'test' not in self.dataset:
#             return None
#         batch_size = self.eval_batch_size if self.eval_batch_size is not None else self.batch_size
#         return DataLoader(
#             self.dataset['test'], 
#             batch_size=batch_size, 
#             collate_fn=self.collate_fn,
#             shuffle=self.shuffle_eval,
#             **self.loader_kwargs
#         )
    
# class QuestionAnsweringDataModule(DataModule):
#     """
#     Base class for Question Answering. Should work for both IR and RC.
    
#     The core idea is that it relies on a Knowledge Base (KB) 
#     to retrieve relevant and irrelevant passages for the questions in the dataset.
        
#     We need to create the batch of questions and passages on-the-fly
#     The inputs should be shaped like (N * M, L), where:
#         * N - number of distinct questions (equal to the batch size)
#         * M - number of passages per question in a batch
#         * L - sequence length

#     Parameters
#     ----------
#     *args, **kwargs: additionnal arguments are passed to DataModule
#     kb: str
#         path towards the knowledge base (Dataset) used to get the passages    
#     image_kb: str, optional
#         Path to the KB that holds pre-computed image features
#         Can be mapped from kb using kb['index']
#     search_key: str, optional
#         This column in the dataset suffixed by '_indices' and '_scores' should hold the result of information retrieval
#         used during evaluation (e.g. the output of ir.search)
#         Suffixed by "_provenance_indices" and "_irrelevant_indices" it should hold:
#             1. the union of relevant search and provenance_indices
#             2. irrelevant results from the search
#         used during training (according to M and n_relevant_passages)
#         Defaults to 'search'
#     filter_train_rels: bool, optional     
#     keep_kb_columns: list, optional
#         Keep only these features in kb and image_kb (defaults to keep everything)
#     kb_format, image_kb_format: dict, optional
#         see Dataset.set_format
#         Overrides keep_kb_columns.
#     kb_input_key: str, optional
#         Defaults to 'passage'
#     """
#     def __init__(self, *args, kb, image_kb=None, search_key='search', 
#                  filter_train_rels=False, keep_kb_columns=None, 
#                  kb_format=None, image_kb_format=None, kb_input_key='passage', **kwargs):
#         super().__init__(*args, **kwargs)
#         #TODO wiki.set_format('torch', ['clip-RN50'])
#         self.kb = verbose_load_from_disk(kb)
#         if kb_format is not None:
#             self.kb.set_format(**kb_format)
#         elif keep_kb_columns is not None:
#             keep_kb_columns = set(keep_kb_columns)
#             self.kb = keep_columns(self.kb, keep_kb_columns)
#         if image_kb is not None:
#             self.image_kb = verbose_load_from_disk(image_kb)            
#             self.padding_passage = [{self.kb_input_key: ''}]
#             if image_kb_format is not None:
#                 self.image_kb.set_format(**image_kb_format)
#             elif keep_kb_columns is not None:
#                 self.image_kb = keep_columns(self.image_kb, keep_kb_columns)
#         else:
#             self.image_kb = None
#             self.padding_passage = ['']
#         self.search_key = search_key    
#         if self.image_formatter.precomputed:
#             self.add_image = self.add_image_features
#         else:
#             self.add_image = self.add_image_path
#         self.filter_train_rels = filter_train_rels
#         self.kb_input_key = kb_input_key
        
#     def setup(self, stage=None):
#         super().setup(stage=stage)
#         if self.filter_train_rels and 'train' in self.dataset:
#             self.filter_rels('train')

#     def filter_rels(self, subset='train'):
#         """
#         Filter out questions of the dataset without any relevant passages.
#         """
#         before_len = len(self.dataset[subset])
#         self.dataset[subset] = self.dataset[subset].filter(
#             lambda item: len(item[f"{self.search_key}_provenance_indices"]) > 0, 
#             new_fingerprint=f"{subset}_{self.search_key}_provenance_indices"
#         )
#         after_len = len(self.dataset[subset])
#         print(f"Filtered {subset} dataset with empty '{self.search_key}_provenance_indices' from {before_len} to {after_len} items")
        
#     def get_training_passages(self, item, with_scores=False):
#         """
#         Parameters
#         ----------
#         item: dict
#             item (e.g. question) from self.train_dataset or self.eval_dataset.
#         with_scores: bool, optional
#             Also return the scores corresponding to the passages
#             Defaults to False.
        
#         Returns
#         -------
#         relevant_passages, irrelevant_passages: list[dict]
#             List of relevant and irrelevant passages selected from self.kb
#             according to:
#                 - self.n_relevant_passages
#                 - self.M
#                 - self.search_key
#         relevant_scores: np.ndarray, optional
#             Shape (self.n_relevant_passages, )
#             Returned only if with_scores
#         irrelevant_scores: np.ndarray, optional 
#             Shape (self.M-self.n_relevant_passages, )
#             Returned only if with_scores
#         """
#         assert self.n_relevant_passages <= self.M
#         # get passages from kb wrt search_key
#         relevant_passages, relevant_scores = [], []
#         all_relevant_indices = item[self.search_key+"_provenance_indices"]
#         n_relevant = min(len(all_relevant_indices), self.n_relevant_passages)
#         if n_relevant > 0:
#             i = np.arange(n_relevant)
#             np.random.shuffle(i)
#             relevant_indices = np.array(all_relevant_indices)[i]
#             if with_scores:
#                 relevant_scores = np.array(item[self.search_key+"_provenance_scores"], dtype=np.float32)[i]
#             relevant_passages = self.kb.select(relevant_indices)
#         irrelevant_passages, irrelevant_scores = [], []
#         all_irrelevant_indices = item[self.search_key+"_irrelevant_indices"]
#         n_irrelevant = min(len(all_irrelevant_indices), self.M-self.n_relevant_passages)
#         if n_irrelevant > 0:
#             i = np.arange(n_irrelevant)
#             np.random.shuffle(i)
#             irrelevant_indices = np.array(all_irrelevant_indices)[i]
#             if with_scores:
#                 irrelevant_scores = np.array(item[self.search_key+"_irrelevant_scores"], dtype=np.float32)[i]
#             irrelevant_passages = self.kb.select(irrelevant_indices)
#         elif n_relevant <= 0:
#             warnings.warn(f"Didn't find any passage for question {item['id']}")
        
#         # multimodal vs. text-only mode
#         if self.image_kb is None:
#             if relevant_passages:
#                 relevant_passages = relevant_passages[self.kb_input_key]
#             if irrelevant_passages:
#                 irrelevant_passages = irrelevant_passages[self.kb_input_key]
#         else:
#             relevant_passages = self.add_image(list(relevant_passages))
#             irrelevant_passages = self.add_image(list(irrelevant_passages))     
#         if with_scores:
#             return relevant_passages, irrelevant_passages, relevant_scores, irrelevant_scores
#         else:
#             return relevant_passages, irrelevant_passages

# class BiEncoderDataModule(QuestionAnsweringDataModule): 
#     """
#     Parameters
#     ----------
#     *args, **kwargs: additionnal arguments are passed to QuestionAnsweringDataModule
#     passage_type_ids: bool, optional
#         Pass token_type_ids=1 for passages (see BertTokenizer for details).
#         This might be useful if you use a shared encoder to encode questions and passages.
#         Defaults to False (by default you use different models to encode questions and passages).
#     """
#     def __init__(self, *args, passage_type_ids=False, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.passage_type_ids = passage_type_ids

#     def collate_fn(self, items):
#         """
#         Collate batch so that each question is associate with n_relevant_passages and M-n irrelevant ones.
#         Also tokenizes input strings

#             * N - number of questions in a batch
#             * M - number of passages per questions
#             * d - dimension of the model/embeddings

#         Returns (a dict of)
#         -------------------
#         question_inputs: dict[torch.LongTensor]
#             input_ids: torch.LongTensor
#                 shape (N, L)
#             **kwargs: 
#                 more tensors depending on the tokenizer, e.g. attention_mask
#         context_inputs: dict[torch.LongTensor]
#             input_ids: torch.LongTensor
#                 shape (N*M, L)
#                 The first N rows correspond to the relevant contexts for the N questions
#                 The rest N*(M-1) rows are irrelevant contexts for all questions.
#             **kwargs: 
#                 idem
#         labels: torch.LongTensor
#             shape (N, )
#             Index of the relevant passage in context_inputs.
#             Should be arange(N) except for padding passages.
#         """        
#         assert self.n_relevant_passages == 1
#         n_irrelevant_passages = self.M-self.n_relevant_passages
#         questions, relevant_passages, irrelevant_passages, labels = [], [], [], []
#         for i, item in enumerate(items):
#             relevant_passage, irrelevant_passage = self.get_training_passages(item)
#             if len(relevant_passage) < 1:
#                 relevant_passage = self.padding_passage
#                 labels.append(self.trainer.lightning_module.loss_fct.ignore_index)
#             else:
#                 labels.append(i)
#             if len(irrelevant_passage) < n_irrelevant_passages:
#                 irrelevant_passage.extend(self.padding_passage*(n_irrelevant_passages-len(irrelevant_passage)))
#             questions.append(item[self.input_key])
#             relevant_passages.extend(relevant_passage)
#             irrelevant_passages.extend(irrelevant_passage)

#         # tokenize questions
#         question_inputs_text = self.tokenizer(questions, **self.tokenization_kwargs)
#         # concatenate passages and tokenize
#         all_passages = relevant_passages + irrelevant_passages
#         if self.image_kb is None:
#             all_passages_text = all_passages
#         else:
#             all_passages_text = [p[self.kb_input_key] for p in all_passages]
#         context_inputs_text = self.tokenizer(all_passages_text, **self.tokenization_kwargs)
#         if self.passage_type_ids:
#             context_inputs_text['token_type_ids'][context_inputs_text['attention_mask'].bool()] = 1
        
#         # wrap it up
#         question_inputs = self.image_formatter.format_batch(question_inputs_text, items)
#         context_inputs = self.image_formatter.format_batch(context_inputs_text, all_passages)
#         labels = torch.tensor(labels)
#         batch = dict(question_inputs=question_inputs, context_inputs=context_inputs, labels=labels)
#         return batch