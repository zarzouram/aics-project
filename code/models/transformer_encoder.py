"""Reference:
Sequence-to-Sequence Modeling with nn.Transformer and TorchText
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from layers.text_encoding import TextEncoding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    def __init__(self,
                 caption_embs_size,
                 num_head=4,
                 num_layers=4,
                 dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        self.model_type = 'TransformerEncoder'

        # Embedding layer
        self.emed_size = caption_embs_size
        self.embedding = TextEncoding(caption_embs_size)

        self.pos_encoder = PositionalEncoding(caption_embs_size, dropout)

        # encoder
        encoder_layers = TransformerEncoderLayer(d_model=caption_embs_size,
                                                 nhead=num_head,
                                                 dim_feedforward=200)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).T
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def forward(self, scr, scr_mask):
        scr = self.embedding(
            input_ids=scr, attention_mask=scr_mask) * math.sqrt(self.emed_size)
        scr = scr.permute(1, 0, 2)  # (B, T, e)  ==> (T, B, e)
        scr = self.pos_encoder(scr)

        mask = self.generate_square_subsequent_mask(scr.size(0), scr.device)
        output = self.transformer_encoder(scr, mask)
        output = output.mean(0)
        return output
