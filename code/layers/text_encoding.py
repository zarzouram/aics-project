from typing import List

import torch
from torch import Tensor
from torch import nn

from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings


class TextEncoding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        """
        Parameters
        ----------
        model_name:     string
                        Bert model absolute directory
        encoder_size:   int
                        Size of sentence encoding
        """
        super(TextEncoding, self).__init__()

        self.embeddings = TransformerWordEmbeddings("distilbert-base-uncased",
                                                    layers="-1,-2",
                                                    fine_tune=True,
                                                    layer_mean=False)

        input_size = self.embeddings.embedding_length
        self.dropout = nn.Dropout(0.5)
        self.reduce = nn.Sequential(nn.Linear(input_size, input_size // 3),
                                    nn.ReLU(),
                                    nn.Linear(input_size // 3, hidden_size),
                                    self.dropout)

    def forward(self, texts: List[Sentence]) -> Tensor:
        """
        Encode captions text using bert + linear MLP.

        Parameters
        ----------
        Texts
        """

        self.embeddings.embed(texts)
        # get bert embeddings
        all_embs = []
        for sent in texts:
            embs_sent = []
            for token in sent:
                embs_sent.append(token.embedding.view(-1, 2))
            embs_sent = torch.stack(embs_sent)
            embs_sent = embs_sent.mean(dim=0)  # mean pool
            all_embs.append(embs_sent)

        all_embs = torch.stack(all_embs)
        all_embs = all_embs.view(all_embs.size(0), -1)

        return self.reduce(all_embs)
