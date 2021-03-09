from typing import List

import torch
from torch import Tensor
from torch import nn

from transformers import DistilBertModel


class TextEncoding(nn.Module):
    def __init__(self, hidden_size: int, lstm_num_layers: int,
                 lstm_bdir: bool) -> None:
        """
        Parameters
        ----------
        model_name:     string
                        Bert model absolute directory
        encoder_size:   int
                        Size of sentence encoding
        """
        super(TextEncoding, self).__init__()

        self.embeddings = DistilBertModel.from_pretrained(
            'distilbert-base-uncased')
        input_size = self.embeddings.embedding_length


        self.reduce = nn.Linear(input_size, hidden_size)

    def forward(self, texts: List[Sentence]) -> Tensor:
        """
        Encode captions text using bert + linear MLP.

        Parameters
        ----------
        Texts
        """




        return self.reduce(text_enc_pooled)
