from typing import List, Optional, Dict

import os
import pathlib

import pickle
import h5py
import json

from collections import OrderedDict

import numpy as np
import torch
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab, GloVe


class SlimDataset(torch.utils.data.Dataset):
    """ SlimDataset class for SLIM.

        Args:
        root_dir:       str
                        Path to root directory.
    """

    def __init__(
        self,
        root_dir: str,
        glove_dir: str,
        glove_name: str,
        glove_dim: int,
        vocab_specials: Dict[str, str],
        vocab_min_freq: int = 2,
        transform=None,
        pretrain: Optional[str] = None,
        vocab_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.images = None
        self.views = None
        self.texts = None
        self.vocab = None
        self.tokens = None
        self.max_len = None
        self.lengths = None

        self.glove_name = glove_name
        self.glove_dir = str(pathlib.Path(os.path.expanduser(glove_dir)))
        self.glove_dim = glove_dim

        self.pretrain = pretrain

        self.root_dir = pathlib.Path(os.path.expanduser(root_dir))
        images_path = self.root_dir / "images.hdf5"

        self.train = True
        if self.root_dir.name != "train":
            self.train = False

        self.vocab_path = vocab_path

        self.vocab_specials = vocab_specials
        self.vocab_min_freq = vocab_min_freq

        if pretrain is None or pretrain == "draw":
            with h5py.File(images_path) as h5_file:
                images_ds = h5_file["images"]
                views_ds = h5_file["cameras"]
                group_name, = list(images_ds.keys())

                # batch size = b
                # n = number of viwes per scene
                # c, h, w = image shape
                self.images = np.array(images_ds[group_name])
                self.views = np.array(views_ds[group_name])
                self.transform = transform

            if pretrain == "draw":
                b, n, c, h, w = self.images.shape
                self.images = self.images.reshape(b * n, c, h, w)
                self.views = self.views.reshape(b * n, -1)

        if pretrain is None or pretrain == "caption_encoder":
            self.get_texts()
            self.build_vocab()
            self.get_tokens()
            self.pad_value = self.vocab[self.vocab_specials["pad"]]

    def initialize(self):
        pass

    def __len__(self) -> int:
        return len(
            self.tokens) if self.images is None else self.images.shape[0]

    def __getitem__(self, i: int):

        if self.images is not None:
            views = torch.as_tensor(self.views[i], dtype=torch.float)
            views = torch.round(views, decimals=3)
            images = torch.as_tensor(self.images[i], dtype=torch.float)
            if self.transform:
                images = self.transform(images)
            if self.tokens is None:
                return i, images, views

        if self.tokens is not None:
            padding_value = self.vocab[self.vocab_specials["pad"]]
            scene_tokens = pad_sequence(self.tokens[i],
                                        padding_value=padding_value)
            length = self.lengths[i]
            if self.images is None:
                return i, scene_tokens, length

        return i, images, views, scene_tokens, length

    def collate_fn(self, batch):
        data = list(zip(*batch))
        idxs = torch.LongTensor(data[0])
        if self.tokens is not None:
            padding_value = self.vocab[self.vocab_specials["pad"]]
            lengths = torch.LongTensor(data[-1])
            tokens = pad_sequence(data[-2],
                                  padding_value=padding_value,
                                  batch_first=True)
            if self.images is None:
                return idxs, tokens, lengths

        if self.images is not None:
            images = torch.stack(data[1])
            views = torch.stack(data[2])

            if self.tokens is None:
                return idxs, images, views, None, None

        return idxs, images, views, tokens, lengths

    def get_texts(self):
        text_pickle = self.root_dir / "texts.pickle"
        with open(text_pickle, "rb") as pickle_file:
            texts = pickle.load(pickle_file)  # type: List[str]
            self.texts = np.array(texts, dtype=object)

    def get_tokens(self):
        tokens_file = self.root_dir / "tokens.json"
        with open(tokens_file) as tokens_json:
            tokens = json.load(tokens_json)  # type: List[List[List[str]]]

        self.tokens = []
        self.lengths = []
        self.max_len = 0
        for scene_tokens in tokens:
            token_ids, tokens_len = [], []
            for view_tokens in scene_tokens:
                token_ids.append(torch.LongTensor(self.vocab(view_tokens)))
                tokens_len.append(len(view_tokens))
                self.max_len = max(self.max_len, tokens_len[-1])
            self.tokens.append(token_ids)
            self.lengths.append(tokens_len)

    def build_vocab(self):
        if self.vocab_path is None:
            if self.train:
                bow_json = self.root_dir / "bow.json"
            else:
                bow_json = self.root_dir.parent / "train/bow.json"

            with open(bow_json) as bow_fjson:
                bow: Dict[str, int] = json.load(bow_fjson)

            bow = OrderedDict(
                sorted(bow.items(), key=lambda x: x[1], reverse=True))
            self.vocab = vocab(bow,
                               min_freq=self.vocab_min_freq,
                               specials=list(self.vocab_specials.values()))
            self.vocab.set_default_index(
                self.vocab[self.vocab_specials["unk"]])

        else:
            self.vocab = torch.load(self.vocab_path)

    def get_glove(self, pad_token: str):
        tokens = list(self.vocab.get_stoi().keys())
        glove_vectors = GloVe(name=self.glove_name,
                              dim=self.glove_dim,
                              cache=self.glove_dir)
        glove_vocab = glove_vectors.stoi
        vectors = []
        for token in tokens:
            if token == pad_token:
                vectors.append(torch.zeros(self.glove_dim))
                continue

            idx = glove_vocab.get(token)
            if idx is None:  # OOV
                vectors.append(
                    xavier_uniform_(torch.ones(1, self.glove_dim)).view(-1))
            else:
                vectors.append(glove_vectors.vectors[idx])

        return torch.stack(vectors)


if __name__ == "__main__":
    from torch.utils.data import DataLoader, RandomSampler
    from tqdm import tqdm

    ds_dir = "/srv/data/zarzouram/lt2318/slim/turk_torch"
    splits = ["test", "train", "val"]
    pretrains = [None, "draw", "caption_encoder"]
    for split in splits:
        for pretrain in pretrains:
            train_ds = SlimDataset(
                root_dir=f"{ds_dir}/{split}",
                glove_name="42B",
                glove_dir="/srv/data/zarzouram/resources/embeddings/Glove/",
                glove_dim=300,
                pretrain=pretrain,
                vocab_specials={
                    "pad": "<pad>",
                    "eos": "<eos>",
                    "unk": "<unk>"
                })
            if split == "train":
                sampler = RandomSampler(train_ds, num_samples=3200)
            else:
                sampler = None
            train_iter = DataLoader(train_ds,
                                    collate_fn=train_ds.collate_fn,
                                    batch_size=32,
                                    shuffle=False,
                                    sampler=sampler)
            counter = 0
            for k in tqdm(train_iter):
                counter += 1

            print(counter)
