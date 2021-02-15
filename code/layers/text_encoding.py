from typing import Tuple
import pathlib as plib
from transformers import BertModel

import torch
from torch import Tensor
from torch import nn


class CaptionEncoding(nn.Module):
    def __init__(self, model_dir: str, encoder_size: int) -> None:
        """
        Parameters
        ----------
        model_name:     string
                        Bert model absolute directory
        encoder_size:   int
                        Size of sentence encoding
        """
        super(CaptionEncoding, self).__init__()

        model_dir = plib.Path(model_dir).expanduser()
        self.bert = BertModel.from_pretrained(model_dir)
        # encoding the last 4 hidden layers of bert (Dim reduction)
        encoder_input_size = 4 * self.bert.config.hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_size, 1024),
            nn.Linear(1024, 265),
            nn.Linear(265, encoder_size),
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                token_type_ids: Tensor) -> Tensor:
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

        token_type_ids: torch.LongTensor
                        of shape (batch_size, sequence_length)
                        Segment token indices.
                        `token_type_ids` from bert tokenizer

        See this link for more information about the parameters
        https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel.forward
        """

        # Sizes:
        # ------
        # batch_size: B
        # sequence_length: T
        # bert hidden layer size: H
        # Encoder output size: E
        output_bert = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                output_hidden_states=True,
                                return_dict=True)

        # Get last 4 hidden layers, concat them and perform mean pooling over
        # sequence
        hidden_states = output_bert.hidden_states[-4:]  # type: Tuple[Tensor]
        text_rep = torch.cat(hidden_states, dim=-1)  # (B,T,4*H) - type: Tensor
        text_rep_pooled = text_rep.mean(dim=1)  # (B, 4*H)

        encoded = self.encoder(text_rep_pooled)  # (B, E)

        return self.dropout(encoded)
