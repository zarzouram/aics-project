import pathlib
import gzip

import numpy as np
import torch

from dataset.classes_helper import BertPreprocessing


class SlimDataset(torch.utils.data.Dataset):
    """ SlimDataset class for SLIM.
        SlimDataset class loads `*.pt.gz` data files. Each `*.pt.gz` file
        includes list of tuples, and these are rearanged to mini batches.

        Args:
        root_dir:       str
                        Path to root directory.
        batch_size:     int
                        Batch size.
        vocab_builder:  Vocabulary builder
        Returns:        None
    """
    def __init__(
        self,
        root_dir: str,
        bert_model_path: str,
        save_tokens_path: str,
        minibatch_size: int,
        train: bool,
    ) -> None:
        super().__init__()

        self.tokenizer = BertPreprocessing(model_path=bert_model_path,
                                           train=train)
        self.__np2str = np.vectorize(lambda x: str(x))

        files = pathlib.Path(root_dir).expanduser().glob("*.pt.gz")
        self.record_list = sorted(files)
        self.train = train
        self.minibatch_size = minibatch_size

    def __len__(self) -> int:
        """Returns number of files and directories in root dir.

        Args: None

        Returns:
            len:    int
                    Number of objects in root dir.
        """

        return len(self.record_list)

    def __getitem__(self, index: int) -> list:
        """Loads data file and returns data with specified index.
        This method reads `<index>.pt.gz` file which includes a list of tuples
        `(images, viewpoints, topdown, captions, *)`, and returns list of
        tuples of tensors `(images, viewpoints, captions)`.
        * Image size: `(b, m, 3, 64, 64)`
        * Viewpoints size: `(b, m, 4)`
        * Captions size: `(b, m, l)`
        Args:
            index (int): Index number.
        Returns:
            data_list (torch.Tensor): List of tuples of tensors
                `(images, viewpoints, captions)`. Length of list is
                `data_num // batch_size`.
        """

        with gzip.open(self.record_list[index], "rb") as f:
            dataset = torch.load(f)

        # number of samples/batch_size:             B
        # mini batch size:                          mB
        # number of (image, captions and views):    N = 10
        # length of caption text:                   l
        # Max length of caption:                    SEQ

        # image size: Tensor(B, N, 3, 64, 64)
        # views size: Tensor(B, N, 4)
        # texts size: List(B, np.array(N, l))
        # captions size: Tensor(B, N, SEQ)
        # image, views, texts, captions = dataset
        image, views, texts = dataset

        # Select rendom image and its viepoint
        # random selected images: (B, 3, 64, 64)
        # Viewpoints of the random selected images: (B, 4)
        # Viewpoints of the other images: (S, 9, 4)
        # captions for the other images: (S, 9, SEQ), two types of captions

        # Generate indices to select random image idx_r
        # idx_r: a uniform random sample from (0, N) of size B
        B = image.size(0)
        N = image.size(1)
        idx_r = np.random.choice(N, B)  # (B)
        imgr = image[torch.arange(B), idx_r, :]  # (B, 3, 64, 64)
        views_imgr = views[torch.arange(B), idx_r]  # (B, 4)

        # Generates indices for all other data except for the random selected
        # images of size (B*(N-1)).
        idx2 = np.tile(np.arange(N), (B, 1))
        idx2 = idx2[idx2 != idx_r[:, None]]
        idx1 = np.repeat(np.arange(B), N - 1)
        views_other = views[idx1, idx2, :].view(B, N - 1, -1)  # (B, 9, 4)

        # (B, 9, SEQ)
        # captions_other = captions[idx1, idx2, :].view(B, N - 1, -1)
        texts_other = texts[idx1, idx2].reshape([B, N - 1])  # (B, 9, l)
        tests_other_list = self.__np2str(
            texts_other.reshape(-1)).tolist()  # (b*9)
        # bert preprocessing
        tokens_data = self.tokenizer.tokenize(tests_other_list)
        tokens_id = tokens_data["input_ids"].view(B, N - 1, -1)
        tokens_type_id = tokens_data["token_type_ids"].view(B, N - 1, -1)
        attention_mask = tokens_data["attention_mask"].view(B, N - 1, -1)

        # Mini batch
        if self.train:
            # size B' = (mB * number of mini batches) < B
            minibatch_num = B // self.minibatch_size
            actual_B = self.minibatch_size * minibatch_num
            # Generate indices to select random samples out of B
            # a uniform random sample from (0 to B) of size B'
            idx_r = np.random.choice(B, actual_B, replace=False)  # (B')

            imgr = imgr[idx_r, :]  # (B', 3, 64, 64)
            views_imgr = views_imgr[idx_r, :]  # (B', 4)
            views_other = views_other[idx_r, :]  # (B', 9, 4)
            # captions_other = captions_other[idx_r, :]  # (B', 9, SEQ)
            tokens_id = tokens_id[idx_r, :]  # (B', 9, SEQ)
            tokens_type_id = tokens_type_id[idx_r, :]  # (B', 9, SEQ)
            attention_mask = attention_mask[idx_r, :]  # (B', 9, SEQ)

            _, *i_dims = imgr.size()
            _, *vi_dims = views_imgr.size()
            _, *vo_dims = views_other.size()
            # _, *c_dims = captions_other.size()
            _, *td_dims = tokens_id.size()

            # Resize: break dim=0 B' into dim0=number of mini batches , dim1=mB
            imgr = imgr.contiguous().view(minibatch_num, self.minibatch_size,
                                          *i_dims)
            views_imgr = views_imgr.contiguous().view(minibatch_num,
                                                      self.minibatch_size,
                                                      *vi_dims)
            views_other = views_other.contiguous().view(
                minibatch_num, self.minibatch_size, *vo_dims)

            # captions_other = captions_other.contiguous().view(
            #     minibatch_num, self.minibatch_size, *c_dims)

            tokens_id = tokens_id.contiguous().view(minibatch_num,
                                                    self.minibatch_size,
                                                    *td_dims)
            tokens_type_id = tokens_type_id.contiguous().view(
                minibatch_num, self.minibatch_size, *td_dims)
            attention_mask = attention_mask.contiguous().view(
                minibatch_num, self.minibatch_size, *td_dims)

            data_list = []
            for i in range(minibatch_num):
                data_list.append([
                    # imgr[i], views_imgr[i], views_other[i],
                    # captions_other[i],
                    imgr[i],
                    views_imgr[i],
                    views_other[i],
                    tokens_id[i],
                    tokens_type_id[i],
                    attention_mask[i],
                ])

        else:
            data_list = [
                # imgr, views_imgr, views_other, captions_other, texts_other
                imgr,
                views_imgr,
                views_other,
                tokens_id,
                tokens_type_id,
                attention_mask,
            ]

        return data_list
