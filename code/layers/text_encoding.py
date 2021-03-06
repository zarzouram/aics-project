from typing import List

import torch
from torch import Tensor
from torch import nn

from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings


class TransormerEmbedding(nn.Module):
    def __init__(self) -> None:
        """
        Parameters
        ----------
        model_name:     string
                        Bert model absolute directory
        encoder_size:   int
                        Size of sentence encoding
        """
        super(TransormerEmbedding, self).__init__()

        self.embeddings = TransformerWordEmbeddings("bert-base-uncased",
                                                    layers="-1,-2,-3,-4",
                                                    fine_tune=False,
                                                    layer_mean=True)

    def forward(self, texts: List[Sentence]) -> Tensor:
        """
        Encode captions text using bert + linear MLP.

        Parameters
        ----------
        Texts
        """

        self.embeddings.embed(texts)
        embs = torch.stack([sent[0].embedding for sent in texts])

        return embs
