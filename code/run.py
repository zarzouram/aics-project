# %%
import torch
from torch import optim
# from torch import Tensor
from torch.utils.data import DataLoader

from custom_classes.gpu_cuda_helper import get_gpu_memory
from dataset.dataset_batcher import SlimDataset
from models.SLIM import SLIM

# %%
# Select cuda device based on the free memory
# Most of the time some cudas are busy but available
memory_usage = get_gpu_memory()
memory_usage_percnt = [m / 11178 for m in memory_usage]
min_memory_usage = min(memory_usage_percnt)
cuda_id = memory_usage_percnt.index(min_memory_usage)
print(f"min memory usage: {min_memory_usage * 100:.2f}%")
if min_memory_usage < 0.5:
    device = torch.device(f"cuda:{cuda_id}")
else:
    device = torch.device("cpu")

print(f"selected device:  {device}")
torch.cuda.empty_cache()

# %%
dataset_dir = "/home/guszarzmo@GU.GU.SE/Corpora/slim/turk_data_torch/"
bert_model_dir = "/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data/bert_data/based_uncased"  # noqa: E501

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

MINI_BATCH = 32
CAPTION_ENC_SZ = 64
VIEWS_ENC_SZ = 32
SC_r_SZ = 265
ITER_NUM = 12
N = 2
DRAW_ENC_SZ = 128
DRAW_DEC_SZ = 128
DRAW_Z_SZ = 3

# %%

train_dataset = SlimDataset(root_dir=dataset_dir + "train",
                            bert_model_path=bert_model_dir,
                            save_tokens_path="",
                            minibatch_size=MINI_BATCH,
                            train=True)

val_dataset = SlimDataset(root_dir=dataset_dir + "valid",
                          bert_model_path=bert_model_dir,
                          save_tokens_path="",
                          minibatch_size=no_mini_batch,
                          train=False)

test_dataset = SlimDataset(root_dir=dataset_dir + "test",
                           bert_model_path=bert_model_dir,
                           save_tokens_path="",
                           minibatch_size=no_mini_batch,
                           train=False)

train_iter = DataLoader(train_dataset,
                        batch_size=file_batch,
                        shuffle=True,
                        num_workers=4)

val_iter = DataLoader(val_dataset,
                      batch_size=file_batch,
                      shuffle=True,
                      num_workers=4)

test_iter = DataLoader(test_dataset,
                       batch_size=file_batch,
                       shuffle=True,
                       num_workers=4)

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

model = SLIM(model_parameters).to(device)

# %%


def lr_decay(global_step: int, lr_init=lr_init, lr_min=lr_min, m=decay_rate):
    lr = -m * global_step + lr_init
    return lr


lr0 = lr_decay(0)
optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: lr_decay(step) / lr0)

# %%
for data in train_iter:
    for batch_data in data:
        pass

# %%
