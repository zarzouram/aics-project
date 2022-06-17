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


# class collate_fn(object):

#     def __init__(self, max_len, pad_id=0):
#         self.max_len = max_len
#         self.pad = pad_id

#     def __call__(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
#         """
#         Padds batch of variable lengthes to a fixed length (max_len)
#         """
#         X, y, ls = zip(*batch)
#         X: Tuple[Tensor]
#         y: Tuple[Tensor]
#         ls: Tuple[Tensor]

#         # pad tuple
#         # [B, max_seq_len, captns_num=5]
#         ls = torch.stack(ls)  # (B, num_captions)
#         y = pad_sequence(y, batch_first=True, padding_value=self.pad)

#         # pad to the max len
#         pad_right = self.max_len - y.size(1)
#         if pad_right > 0:
#             # [B, captns_num, max_seq_len]
#             y = y.permute(0, 2, 1)  # type: Tensor
#             y = ConstantPad1d((0, pad_right), value=self.pad)(y)
#             y = y.permute(0, 2, 1)  # [B, max_len, captns_num]

#         X = torch.stack(X)  # (B, 3, 256, 256)

#         return X, y, ls
