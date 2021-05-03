from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from transformers import DistilBertModel


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

        self.embeddings = DistilBertModel.from_pretrained(
            'distilbert-base-uncased')
        input_size = 2 * self.embeddings.config.hidden_size

        self.encoder = nn.Linear(input_size, hidden_size)

        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Encode captions text using bert + linear MLP.
        Parameters
        ----------
        input_ids:      torch.LongTensor
                        of shape (batch_size, sequence_length)
                        token id in the vocabulary. `input_ids` from bert
                        tokenizer.
        attention_mask: torch.FloatTensor
                        of shape (batch_size, sequence_length)
                        Padding token mask to avoid performing attention them.
                        `attention_mask` from bert tokenizer
        See this link for more information about the parameters
        https://huggingface.co/transformers/model_doc/distilbert.html#distilbertmodel
        """

        # Sizes:
        # ------
        # batch_size: B
        # sequence_length: T
        # bert hidden layer size: H
        # Encoder output size: E

        # bert model accepts tensors in shape of (batch_size, sequence_length),
        # refrom the sizes from (B, N, T) to (B*N, T)
        seq_len = input_ids.size(-1)
        output_bert = self.embeddings(input_ids=input_ids.view(-1, seq_len),
                                      attention_mask=attention_mask.view(
                                          -1, seq_len),
                                      output_hidden_states=True,
                                      return_dict=True)

        # Get last 4 hidden layers, concat them and perform mean pooling over
        # sequence
        hidden_states = output_bert.hidden_states[-2:]  # type: Tuple[Tensor]
        text_rep = torch.cat(hidden_states, dim=-1)  # (B,T,2*H) - type: Tensor
        text_rep_pooled = text_rep.mean(dim=1)  # (B, 2*H)

        encoded = nn.tansh(self.encoder(text_rep_pooled))  # (B, E)

        return self.dropout(encoded)
