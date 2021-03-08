# %%
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim

import flair

from dataset.dataset_batcher import SlimDataset
from models.SLIM import SLIM

from dataset.preprocessing import get_mini_batch
from helpers.train_helper import Trainer
from helpers.early_stopping import EarlyStopping
from helpers.scheduler import LinearDecayLR

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
lr_final = 5e-5
step_num = 1.5e4

CAPTION_ENC_SZ = 64
VIEWS_ENC_SZ = 32
SC_r_SZ = 256
ITER_NUM = 12
N = 2
DRAW_ENC_SZ = 128
DRAW_DEC_SZ = 128
DRAW_Z_SZ = 64

MINI_BATCH_SZ = 32
SAMPLE_NUM = 3200
EPOCH_STEP = int(SAMPLE_NUM / MINI_BATCH_SZ)
CHECK_POINT = EPOCH_STEP * 5

# %%
# Select cuda device based on the free memory
# Most of the time some cudas are busy but available
# torch.cuda.empty_cache()

from utils.gpu_cuda_helper import get_gpus_avail

cuda_idx = get_gpus_avail()
device = None
if not cuda_idx:
    device = torch.device("cpu")
elif len(cuda_idx) >= 1:
    device = torch.device(f"cuda:{cuda_idx[0][0]}")
    # if len(cuda_idx) != 1:
    #     cuda_idx = [i for i, _ in cuda_idx]
    #     print(f"Parallel Mode, cuda ids are: {cuda_idx}")

device = torch.device("cuda:2")
print(f"\ndevice selected: {device}")
flair.device = device

# %%
from utils.visualization import Visualizations

# Visualization
vis = Visualizations()

# %%


def custom_collate(data):
    return data


train_dataset = SlimDataset(root_dir=dataset_dir + "train")

val_dataset = SlimDataset(root_dir=dataset_dir + "valid")

test_dataset = SlimDataset(root_dir=dataset_dir + "test")

train_iter = DataLoader(train_dataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=device.type == "cuda",
                        collate_fn=custom_collate)

val_iter = DataLoader(val_dataset,
                      batch_size=1,
                      shuffle=True,
                      num_workers=2,
                      pin_memory=device.type == "cuda",
                      collate_fn=custom_collate)

test_iter = DataLoader(test_dataset,
                       batch_size=file_batch,
                       shuffle=True,
                       num_workers=2,
                       collate_fn=custom_collate)

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
model = model.to(device)
# if cuda_idx:
#     model = nn.DataParallel(model, cuda_idx)

# %%

optimizer = optim.Adam(model.parameters(), lr=lr_init)
scheduler = LinearDecayLR(optimizer)
# scheduler = LinearDecayLR(optimizer,
#                           lr_i=lr_init,
#                           lr_f=lr_final,
#                           s_n=step_num)

# %%

print()
slim_train = Trainer(model,
                     device,
                     epoch_interval=EPOCH_STEP,
                     save_path=model_path,
                     check_model_size=False)

# print(f"Model memory Size: {slim_train.model_memory_gb:.2f} GB")

# %%
# early stopping
es = EarlyStopping(patience=3, min_delta=0.1)

# %%
while slim_train.in_train:
    slim_train.local_steps = 0  # 1 epoch = SAMPLE_NUM local steps
    slim_train.train_loss = 0

    # train
    with tqdm(train_iter, leave=False, unit="file") as train_pb:
        # load one file (max 64 samples per file)
        for train_batch in train_pb:
            # progress bar one step
            train_pb.set_description(f"LocalStep {slim_train.local_steps}")

            trn_mini_b = get_mini_batch(data=train_batch[0],
                                        size_=MINI_BATCH_SZ)

            with tqdm(trn_mini_b, leave=False, unit="minibatch") as minipb:
                minipb.set_description("minibatch train")

                # train min batches
                for data in minipb:
                    slim_train.train(model, optimizer, scheduler, data)

                    # update progress bars
                    minipb.set_postfix({"train/loss": slim_train.train_loss})
                    slim_train.postfix["train/loss"] = slim_train.train_loss
                    slim_train.trainpb.set_postfix(slim_train.postfix)

                    # eval, each CHECK_POINT steps (every 5 epochs)
                    if slim_train.global_steps % (CHECK_POINT - 1) == 0:
                        model_tested = True
                        slim_train.val_loss = 0
                        slim_train.val_steps = 0
                        for val_batch in val_iter:
                            val_mini_batches = get_mini_batch(
                                data=val_batch[0], size_=MINI_BATCH_SZ)
                            slim_train.eval(model, val_mini_batches)

                        slim_train.val_loss = \
                            slim_train.val_loss / slim_train.val_steps

                        # update main progress bar
                        slim_train.postfix["test/loss"] = slim_train.val_loss
                        slim_train.trainpb.set_postfix(slim_train.postfix)

                        # plot validation
                        vis.plot_loss(slim_train.val_loss,
                                      slim_train.global_steps + 1,
                                      "Validation")

                        # save model
                        if slim_train.val_loss > slim_train.best_loss:
                            slim_train.best_loss = slim_train.val_loss
                            slim_train.save_checkpoint(model, optimizer,
                                                       scheduler)

                        # early stopping
                        if es.step(slim_train.val_loss):
                            slim_train.train_loss = \
                                slim_train.train_loss / slim_train.local_steps
                            slim_train.in_train = False

                    # plot
                    if slim_train.global_steps % 3 == 0:
                        vis.plot_loss(slim_train.train_loss,
                                      slim_train.global_steps + 1, "Train")

                    if slim_train.epoch_finished or not slim_train.in_train:
                        break

            if slim_train.epoch_finished or not slim_train.in_train:
                slim_train.epoch_finished = False
                print("Training finished ...")
                break

# %%
print("Done")
