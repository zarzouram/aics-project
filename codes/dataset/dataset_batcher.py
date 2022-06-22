from typing import Optional

import os
import h5py
import pathlib

import numpy as np
import torch


class SlimDataset(torch.utils.data.Dataset):
    """ SlimDataset class for SLIM.

        Args:
        root_dir:       str
                        Path to root directory.
    """

    def __init__(
            self,
            root_dir: str,
            pretrain: Optional[str],  # draw or caption_encoder
            transform=None
    ) -> None:
        super().__init__()

        self.pretrain = pretrain
        root_dir = os.path.expanduser(root_dir)
        images_path = pathlib.Path(root_dir) / "images.hdf5"

        with h5py.File(images_path) as h5_file:
            if pretrain is None or pretrain != "caption_encoder":
                images_ds = h5_file["images"]
                views_ds = h5_file["cameras"]
                group_name, = list(images_ds.keys())
                self.images = np.array(images_ds[group_name])
                self.views = np.array(views_ds[group_name])
                self.transform = transform
                if pretrain.find("draw") != -1:
                    self.texts = None
                    self.tokens = None

            if pretrain is None or pretrain.find("draw") == -1:
                pass
                if pretrain == "caption_encoder":
                    self.images = None
                    self.views = None

    def __len__(self) -> int:
        return self.images.shape[0] if self.images else len(self.tokens)

    def __getitem__(self, i: int):

        if self.images:
            views = torch.as_tensor(self.views[i], dtype=torch.float)
            views = torch.round(views, decimals=3)
            images = torch.as_tensor(self.images[i], dtype=torch.float)
            if self.transform:
                images = self.transform(images)
            if self.tokens is None:
                return images, views

        if self.tokens:
            pass
            if self.images is None:
                # return caption
                pass

        return images, views

    def collate_fn(self):
        pass
