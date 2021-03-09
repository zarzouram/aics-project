from typing import List

import torch
from torch import Tensor
from torch import nn

from transformers import DistilBertModel
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,\
#                                pad_sequence

# from flair.data import Sentence
# from flair.embeddings import TransformerWordEmbeddings


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

        # self.embeddings = TransformerWordEmbeddings("distilbert-base-uncased",
        #                                             layers="-1,-2",
        #                                             fine_tune=True,
        #                                             layer_mean=True)

        self.embeddings = DistilBertModel.from_pretrained(
            'distilbert-base-uncased')
        input_size = self.embeddings.embedding_length
        # self.lstm = nn.LSTM(input_size=input_size,
        #                     hidden_size=hidden_size,
        #                     num_layers=lstm_num_layers,
        #                     bidirectional=lstm_bdir,
        #                     batch_first=True)

        self.reduce = nn.Linear(input_size, hidden_size)

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
        lens = []
        for sent in texts:
            embs_sent = []
            lens.append(len(sent))
            for token in sent:
                embs_sent.append(token.embedding)
            embs_sent = torch.stack(embs_sent)
            all_embs.append(embs_sent)
        # padd variable lenghhes
        embs_pad = pad_sequence(all_embs, batch_first=True)
        # # pack sequence -> recurrent network -> unpack sequence
        # embs_packed = pack_padded_sequence(embs_pad,
        #                                    lens,
        #                                    batch_first=True,
        #                                    enforce_sorted=False)
        # lstm_out, _ = self.lstm(embs_packed)
        # seq_unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # mean pooling to get sentence representation
        text_enc_pooled = embs_pad.mean(dim=1)

        return self.reduce(text_enc_pooled)
