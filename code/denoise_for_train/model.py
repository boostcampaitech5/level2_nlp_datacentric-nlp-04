import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


class Transformer(nn.Module):
    '''
    Use only transformer encoder to get embedding vectors
    of the given tokens.
    Use max of every tokens, not only cls token.
    Use FC layer as classifying layer.
    '''
    def __init__(self, num_tokens, dim_model, num_heads, 
                 num_encoder_layers):
        super().__init__()

        # INFO
        self.model_type = 'Transformer'
        self.dim_model = dim_model

        # Layers
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, max_len=200
        )
        self.embedding = nn.Embedding(
            num_tokens, dim_model, padding_idx=10,
            )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model, nhead=num_heads,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers
        )
        self.out = nn.Linear(dim_model, 1)

    def forward(self, src):
        src = src
        # torch.Size([150, 64])
        src = self.embedding(src) * math.sqrt(self.dim_model)
        # torch.Size([150, 64, 256])
        src = self.positional_encoder(src)
        # torch.Size([150, 64, 256])
        src = src.permute(1,0,2)
        # torch.Size([64, 150, 256]) 
        transformer_out = self.transformer(src)
        # torch.Size([64, 150, 256])

        out = torch.max(transformer_out, dim=1)[0]
        # torch.Size([64, 256])
        out = self.out(out)
        # torch.Size([64, 1])
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len, dropout_p=0.1):
        super().__init__()
        # 드롭 아웃
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
    
