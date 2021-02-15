from typing import List, Dict
import pathlib as plib

from torch import Tensor
from transformers import AutoTokenizer


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
    def __init__(self, model_path: str, train: bool, init_files_dir="") -> None:
        model_path = plib.Path(model_path).expanduser()
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                         do_lower_case=False)
        self.train = train

        if init_files_dir and train:
            json_files = plib.Path(init_files_dir).expanduser().glob("*.json")
            text_files = plib.Path(init_files_dir).expanduser().glob("*.text")
            if len(json_files) > 1 and len(text_files) > 1:
                self.load_status(load_dir=init_files_dir)

    def tokenize(self, sentences: List[str]) -> Dict[str, Tensor]:
        encoded_inputs = self.__tokenizer(sentences,
                                          padding='longest',
                                          return_tensors="pt")
        return encoded_inputs

    def save_status(self, save_dir: str):
        self.__tokenizer.save_pretrained(save_dir)

    def load_status(self, load_dir: str):
        self.__tokenizer.from_pretrained(load_dir)
