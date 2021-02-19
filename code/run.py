# %%
import torch
# from torch.nn import DataParallel
from torch.utils.data import DataLoader
# from torch import nn
from helpers.gpu_cuda_helper import get_gpus_avail
from dataset.dataset_batcher import SlimDataset
from models.SLIM import SLIM
from helpers.train_helper import Trainer

# %% [markdown]
# ## Notes:
# Do not forget to run the below command before starting the script
# `python -m visdom.server`

# %%
# source: https://github.com/NVIDIA/framework-determinism/blob/master/pytorch.md

import random
import numpy as np

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
CUDA_LAUNCH_BLOCKING = 1

# %%
dataset_dir = "/home/guszarzmo@GU.GU.SE/Corpora/slim/turk_data_torch/"
bert_model_dir = "/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data/bert_data/based_uncased"  # noqa: E501
model_path = "/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data/models/slim.pt"

# %%
file_batch = 1
no_mini_batch = 0
bert_hidden_size = 768
views_emb_size = 4
image_width = 64
image_height = 64
image_color = 3

lr_init = 5e-4
lr_min = 5e-5
decay_rate = (lr_init - lr_min) / 1e6

CAPTION_ENC_SZ = 64
VIEWS_ENC_SZ = 32
SC_r_SZ = 256
ITER_NUM = 12
N = 2
DRAW_ENC_SZ = 128
DRAW_DEC_SZ = 128
DRAW_Z_SZ = 64
MINI_BATCH = 32

SAMPLE_NUM = 3200


# %%
# Select cuda device based on the free memory
# Most of the time some cudas are busy but available
# torch.cuda.empty_cache()
cuda_idx = get_gpus_avail()
device = None
if not cuda_idx:
    device = torch.device("cpu")
elif len(cuda_idx) >= 1:
    device = torch.device(f"cuda:{cuda_idx[0][0]}")
    # if len(cuda_idx) != 1:
    #     cuda_idx = [i for i, _ in cuda_idx]
    #     print(f"Parallel Mode, cuda ids are: {cuda_idx}")

# device = torch.device("cpu")
print(f"\ndevice selected: {device}")

# %%

train_dataset = SlimDataset(root_dir=dataset_dir + "train")

val_dataset = SlimDataset(root_dir=dataset_dir + "valid")

test_dataset = SlimDataset(root_dir=dataset_dir + "test")

test_iter = DataLoader(test_dataset,
                       batch_size=file_batch,
                       shuffle=True,
                       num_workers=2)

# %%
model_parameters = {
    "bert_model_dir": bert_model_dir,
    "caption_embs_size": CAPTION_ENC_SZ,
    "views_emb_size": views_emb_size,
    "views_enc_size": VIEWS_ENC_SZ,
    "scene_rep_size": SC_r_SZ,
    "image_width": image_width,
    "image_height": image_height,
    "image_color": image_color,
    "iter_num": ITER_NUM,
    "N": N,
    "draw_encoder_size": DRAW_ENC_SZ,
    "draw_decoder_size": DRAW_ENC_SZ,
    "z_size": DRAW_Z_SZ,
}

model = SLIM(model_parameters)
model.to(device)
# if cuda_idx:
#     model = nn.DataParallel(model, cuda_idx)

# %%
optimizer_config = {"decay_rate": decay_rate, "lr_init": lr_init}

slim_train = Trainer(model,
                     device,
                     early_stop_int=500,
                     sample_num=SAMPLE_NUM,
                     opt_config=optimizer_config,
                     save_path=model_path,
                     datasets=[train_dataset, val_dataset])

# %%
while slim_train.in_train:
    slim_train.train_eval(model)


# %%
print("Done")


# %%
