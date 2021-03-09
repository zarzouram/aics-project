from typing import List, Dict
import pathlib as plib

from torch import Tensor
from transformers import DistilBertTokenizerFast


class BertPreprocessing:
    """Word-cocab encoder.
    Args:
        None.

    Attributes:
        word2index: collections.defaultdict
                    Word to index dict
        index2word: collections.defaultdict
                    Index to word dict
        n_words:    int
                    Number of words.
    """
    def __init__(self, ) -> None:
        self.__tokenizer = DistilBertTokenizerFast.from_pretrained(
            'distilbert-base-uncased')

    def tokenize(self, sentences: List[str]) -> Dict[str, Tensor]:
        encoded_inputs = self.__tokenizer(sentences,
                                          padding='longest',
                                          return_token_type_ids=False,
                                          return_tensors="pt")
        return encoded_inputs

    def save_status(self, save_dir: str):
        self.__tokenizer.save_pretrained(save_dir)

    def load_status(self, load_dir: str):
        self.__tokenizer.from_pretrained(load_dir)
