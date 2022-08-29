from typing import List

import copy
from collections import OrderedDict
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    # This class is copied from:
    # https://github.com/pytorch/examples/blob/d5478765d38210addf474dd73faf0d103052027a/word_language_model/model.py#L65-L105
    r"""Inject some information about the relative or absolute position of the
    tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so
        that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :].requires_grad_(False)
        return self.dropout(x)


class EncoderLayer(nn.Module):

    def __init__(self, embd_size: int, feedforward_dim: int, num_heads: int,
                 dropout: float):
        super(EncoderLayer, self).__init__()

        # encoder layer
        self.self_attn = nn.MultiheadAttention(embed_dim=embd_size,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)

        self.ff = nn.Sequential(
            OrderedDict([("ff_linear1", nn.Linear(embd_size, feedforward_dim)),
                         ("ff_activation", nn.ReLU()),
                         ("ff_dropout", nn.Dropout(dropout)),
                         ("ff_linear2", nn.Linear(feedforward_dim,
                                                  embd_size))]))
        self.norm1 = nn.LayerNorm(embd_size)
        self.norm2 = nn.LayerNorm(embd_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        attn_shape = (src.size(1), src.size(1))
        src_mask = torch.triu(torch.full(attn_shape, float("-inf")),
                              diagonal=1)

        x = self.norm1(src)
        x, weights = self.self_attn(x,
                                    x,
                                    x,
                                    attn_mask=src_mask,
                                    need_weights=True,
                                    average_attn_weights=False)
        x = self.dropout1(x)
        x += src
        x += self.dropout2(self.ff(self.norm2(x)))

        return x, weights


class CaptionEncoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embd_size: int,
                 ff_dim: int = 200,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropouts: List[float] = [0.5, 0.1]):
        super(CaptionEncoder, self).__init__()

        # Embedding layer
        self.emed_size = embd_size
        self.embedding = nn.Embedding(vocab_size, embd_size)
        self.pos_encoder = PositionalEncoding(embd_size, dropouts[0])
        encoder_layer = EncoderLayer(embd_size=embd_size,
                                     feedforward_dim=ff_dim,
                                     num_heads=num_heads,
                                     dropout=dropouts[1])
        self.transfomer_encoders = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)])

        self.init_weights()

    def init_weights(self):
        for p in self.transfomer_encoders.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):

        x = self.embedding(x) * math.sqrt(self.emed_size)
        output = self.pos_encoder(x)
        attn_weights = []
        for transfomer_encoder in self.transfomer_encoders:
            output, weights = transfomer_encoder(output)
            attn_weights.append(weights)

        return output, torch.stack(attn_weights, dim=1)
