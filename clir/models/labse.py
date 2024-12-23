import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class LABSEModule(nn.Module):
    def __init__(self):
        super().__init__()
        smodel = SentenceTransformer('sentence-transformers/LaBSE')
        
        self.model = smodel[0].auto_model
        # self.tokenizer = smodel[0].tokenizer
        self.pooler = smodel[1]
        self.ff = smodel[2]
        # self.normalizer = smodel[3]

    def forward(self, input_ids, attention_mask, **kwargs):
        x = self.model(input_ids, attention_mask, token_type_ids=torch.zeros_like(input_ids), return_dict=False)
        features = {
            "token_embeddings": x[0],
            "attention_mask": attention_mask,
            "all_layer_embeddings": x[1]
        }
        self.ff(self.pooler(features))
        return features["sentence_embedding"]
