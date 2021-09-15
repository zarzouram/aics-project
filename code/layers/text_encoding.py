# from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from transformers import DistilBertModel, DistilBertConfig


class TextEncoding(nn.Module):
    def __init__(self, caption_embs_size) -> None:
        """
        Parameters
        ----------
        model_name:     string
                        Bert model absolute directory
        encoder_size:   int
                        Size of sentence encoding
        """
        super(TextEncoding, self).__init__()

        # self.bert_embed = DistilBertModel.from_pretrained(
        #     'distilbert-base-uncased')
        configuration = DistilBertConfig()
        configuration.architectures = "DistilBertModel"
        configuration.sinusoidal_pos_embds = True
        configuration.is_encoder_decoder = True
        self.bert_embed = DistilBertModel(configuration)
        embed_size = 2 * self.bert_embed.config.hidden_size
        self.dropout = nn.Dropout(0.5)
        self.encoder = nn.Sequential(
            nn.Linear(embed_size, embed_size // 3), nn.Tanh(),
            nn.Linear(embed_size // 3, caption_embs_size), self.dropout)

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
        with torch.no_grad():
            hidden_states = self.bert_embed(
                input_ids=input_ids.view(-1, seq_len),
                attention_mask=attention_mask.view(-1, seq_len),
                output_hidden_states=True,
                return_dict=False)[1][-2:]

            # Get last 2 hidden layers, concat them and perform mean pooling
            # over sequence
            text_rep = torch.cat(hidden_states, dim=-1)  # (B,T,2*H)

        encoded = self.encoder(text_rep)  # (B,T,E)

        return encoded
