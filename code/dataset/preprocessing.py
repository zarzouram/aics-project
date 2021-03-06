from typing import List
import numpy as np
import torch
from torch import Tensor
import gc


def split_query_context(data: List[Tensor]) -> List[Tensor]:
    # random selected query images: (B, 10, 3, 64, 64) ==> (B, 3, 64, 64)
    # Select query viewpoint: (B, 10, 4) ==> (B, 4)
    # Context viewpoints: (B, 10, 4) ==> (B, 9, 4)
    # Context captions data: (B, 10, SEQ) ==> (B, 9, SEQ)
    images = torch.squeeze(data[0])
    views = torch.squeeze(data[1])
    text = np.array(data[2], dtype=object)
    # text = text.reshape(views.size(0), views.size(1))

    B = images.size(0)
    N = images.size(1)
    views_context = views.clone()
    # Generate indices to select random query idx_r
    # idx_r: a uniform random sample from (0, N) of size B
    idx_r = np.random.choice(N, B)  # (B)
    image_query = images[torch.arange(B), idx_r, :]  # (B, 3, 64, 64)
    view_query = views[torch.arange(B), idx_r]  # (B, 4)

    # Generates indices for all other data except for the random selected: idx2
    idx2 = np.tile(np.arange(N), (B, 1))
    idx2 = idx2[idx2 != idx_r[:, None]]
    idx1 = np.repeat(np.arange(B), N - 1)
    views_context = views_context[idx1, idx2, :].view(B, N - 1,
                                                      -1)  # (B, 9, 4)
    text_context = text[idx1, idx2].reshape(B, N - 1)  # (B, 9)

    del images, views
    gc.collect()

    return [image_query, view_query, views_context, text_context]


def get_mini_batch(data: List[Tensor], size_: int = 0) -> List:

    # Split data to query and context
    data_splitted = split_query_context(data)

    image_query = data_splitted[0]
    view_query = data_splitted[1]
    views_context = data_splitted[2]
    text_context = data_splitted[3]

    if size_ > 0:
        # size B' = (mB * number of mini batches) < B
        B = image_query.size(0)
        batch_num = B // size_
        actual_B = size_ * batch_num
        # Generate indices to select random samples out of B
        # a uniform random sample from (0 to B) of size B'
        idx_r = np.random.choice(B, actual_B, replace=False)  # (B')

        image_query = image_query[idx_r, :]  # (B', 3, 64, 64)
        view_query = view_query[idx_r, :]  # (B', 4)
        views_context = views_context[idx_r, :]  # (B', 9, 4)
        text_context = text_context[idx_r, :]

        _, *i_dims = image_query.size()
        _, vi_dims = view_query.size()
        _, *vo_dims = views_context.size()

        # Resize: break dim=0 B' into dim0=number of mini batches , dim1=mB
        image_query = image_query.contiguous().view(batch_num, size_, *i_dims)

        view_query = view_query.contiguous().view(batch_num, size_, vi_dims)

        views_context = views_context.contiguous().view(
            batch_num, size_, *vo_dims)

        text_context = text_context.reshape(batch_num, -1)

        data_list = []
        for i in range(batch_num):
            data_list.append([
                image_query[i], view_query[i], views_context[i],
                text_context[i]
            ])
        return data_list

    else:
        return [image_query, view_query, views_context, text_context]
