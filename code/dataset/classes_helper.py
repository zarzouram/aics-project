from typing import DefaultDict
import json
from collections import defaultdict


class Vocabulary:
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
    def __init__(self, PAD=0, SOS=1, EOS=2) -> None:

        self.word2index: DefaultDict[str, int] = defaultdict(int)
        self.index2word: DefaultDict[int, str] =  defaultdict(int)
        self.index2word[PAD] = "PAD"
        self.index2word[SOS] = "SOS"
        self.index2word[EOS] = "EOS"
        self.n_words = 3

        # self.removal = """!@#$%^&*()_-+=|¥~`[]{}:;"',.<>/?"""

    def __len__(self) -> int:
        """Length of registered words."""
        return self.n_words

    def token2index(
        self,
        token: str,
    ) -> int:
        """Convert token to indices list.
        Args:
            token:      str
                        word string.
        Returns:
            indices:    int
                        Indices list.
        """

        if token not in self.word2index:
            self.word2index[token] = self.n_words
            self.index2word[self.n_words] = token
            self.n_words += 1

        return self.word2index.get(token)

    def __getitem__(self, item):
        # Get the word id of given word or vise versa
        if isinstance(item, int):
            return self.index2word.get(item, None)
        else:
            return self.word2index.get(item, None)

    def to_json(self, path: str) -> None:
        """Saves to json file.
        Args:
            path (str): Path to saved file.
        """

        data = {
            "word2index": self.word2index,
            "index2word": self.index2word,
            "n_words": self.n_words,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    def read_json(self, path: str) -> None:
        """Reads saved json file.
        Args:
            path (str): Path to json file.
        """

        with open(path, "r") as f:
            data = json.load(f)

        word2index = {k: int(v) for k, v in data["word2index"].items()}
        self.word2index = defaultdict(int, word2index)
        self.index2word = {int(k): v for k, v in data["index2word"].items()}
        self.n_words = int(data["n_words"])
