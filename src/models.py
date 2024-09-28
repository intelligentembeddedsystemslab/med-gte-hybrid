from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from sentence_transformers import SentenceTransformer

from typing import List


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        # first set hidden states corresponding to padding tokens to 0
        # then average last hidden states of each token in the sequence (ignoring PAD tokens)
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class EmbeddingModel:
    # model_name should either be avilable from huggingface hub or a local folder
    # pooling may be 'cls' or 'mean'
    def __init__(self, model_name = "thenlper/gte-large", pooling = 'mean', prompt=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Model {model_name} initializing on {self.device}')

        self.prompt = prompt
        self.pooling = pooling

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def get_embedding(self, input:List[str]) -> np.ndarray:
        
        if self.prompt is not None:
            input = [self.prompt + text for text in input]

        tokenized_input = self.tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.model(**tokenized_input)
        
        if self.pooling == 'cls':
            embeddings = outputs.last_hidden_state[:,0,:]
        
        if self.pooling == 'mean':
            embeddings = average_pool(outputs.last_hidden_state, tokenized_input['attention_mask'])
        
        # normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()


class HybridModel(SentenceTransformer):
    # Med-gte-hybrid model
    def __init__(self, model_path_1='./models/med_gte_simcse', model_path_2='./models/med_gte_tsdae'):
        super().__init__()
        self.model1 = SentenceTransformer(model_path_1)
        self.model2 = SentenceTransformer(model_path_2)

    def encode(self, sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True, 
               normalize_embeddings=False, **kwargs):
       
        # Get embeddings from both models
        embeddings1 = self.model1.encode(
            sentences, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar, 
            convert_to_tensor=True, 
            device='cuda',
            normalize_embeddings=normalize_embeddings, 
        )

        embeddings2 = self.model2.encode(
            sentences, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar, 
            convert_to_tensor=True,
            device='cuda',
            normalize_embeddings=normalize_embeddings, 
        )

        # Concatenate embeddings along the feature dimension
        combined_embeddings = torch.cat([embeddings1, embeddings2], dim=1)

        if convert_to_numpy:
            return combined_embeddings.cpu().numpy()
        return combined_embeddings


# simple one layer classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_size=1024):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# For convenience predefine some models with short names
# Lambda functions for lazy loading 
modelsdict = {
        'bert': lambda: EmbeddingModel(model_name='bert-base-uncased', pooling="mean"),
        'cbert': lambda: EmbeddingModel(model_name='medicalai/ClinicalBERT', pooling="mean"),
        'gte': lambda: EmbeddingModel(model_name='thenlper/gte-large', pooling="mean"),
        'mxbai': lambda: EmbeddingModel(model_name='mixedbread-ai/mxbai-embed-large-v1', pooling="mean"),
        'bge': lambda: EmbeddingModel(model_name='BAAI/bge-large-en-v1.5', pooling="mean"),
        'uae': lambda: EmbeddingModel(model_name='WhereIsAI/UAE-Large-V1', pooling="mean"),
        'tsdae': lambda: EmbeddingModel(model_name='./models/med_gte_tsdae', pooling='mean'),
        'simcse': lambda: EmbeddingModel(model_name='./models/med_gte_simcse', pooling='mean'),
        'hybrid': lambda: HybridModel(),
        'clf': lambda: SimpleClassifier(input_size=1024),
    }