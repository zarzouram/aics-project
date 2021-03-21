# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
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
#   ## Notes:
#   Do not forget to run the below command before starting the script
#   `python -m visdom.server`
# %% [markdown]
#  ### Imports

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

# %% [markdown]
#  ### Initialize some variables
#

# %%
dataset_dir = "/home/guszarzmo@GU.GU.SE/Corpora/slim/turk_data_torch_flair/"
model_path = "/home/guszarzmo@GU.GU.SE/LT2318/aics-project/data/models/"

file_batch = 1
no_mini_batch = 0
bert_hidden_size = 768
views_emb_size = 4
image_width = 64
image_height = 64
image_color = 3

lr_init = 5e-4
lr_final = 5e-5
step_num = 1e6

CAPTION_ENC_SZ = 64
VIEWS_ENC_SZ = 32
SC_r_SZ = 256
ITER_NUM = 12
DRAW_h_SZ = 128
DRAW_Z_SZ = 3

MINI_BATCH_SZ = 32
SAMPLE_NUM = 3200
EPOCH_STEP = int(SAMPLE_NUM / MINI_BATCH_SZ)
CHECK_POINT = EPOCH_STEP * 5

model_parameters = {
    "caption_embs_size": CAPTION_ENC_SZ,
    "views_emb_size": views_emb_size,
    "views_enc_size": VIEWS_ENC_SZ,
    "scene_rep_size": SC_r_SZ,
    "image_width": image_width,
    "image_height": image_height,
    "image_color": image_color,
    "iter_num": ITER_NUM,
    "draw_h_size": DRAW_h_SZ,
    "z_size": DRAW_Z_SZ,
}

# %% [markdown]
#  ### Auotomatic GPU selection

# %%
# Select cuda device based on the free memory
# Most of the time some cudas are busy but available
# torch.cuda.empty_cache()
use_cpu = False

from utils.gpu_cuda_helper import get_gpus_avail

cuda_idx = get_gpus_avail()
device = None
if not cuda_idx:
    device = torch.device("cpu")
elif len(cuda_idx) >= 1:
    cuda_id = cuda_idx[0][0]
    device = torch.device(f"cuda:{cuda_id}")

if use_cpu:
    device = torch.device("cpu")

print(f"\ndevice selected: {device}")
flair.device = device

# %% [markdown]
#  ### Initialize plotting class

# %%
from utils.visualization import Visualizations

# Visualization
vis = Visualizations()

# %% [markdown]
#  ### DataLoader

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

# %% [markdown]
#  ### Construct the model

# %%
model = SLIM(model_parameters)
model = model.to(device)
# if cuda_idx:
#     model = nn.DataParallel(model, cuda_idx)

# %%

optimizer = optim.Adam(model.parameters(), lr=lr_init)
scheduler = LinearDecayLR(optimizer)

# %% [markdown]
#  ### Model Train

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
check_grad = False
resume = True

if resume:
    model_name = "slim_18-03-08h06_7485.pt"
    load_path = model_path + model_name
    model_data = torch.load(load_path, map_location=device)
    model_state_dict = model_data["model_state_dict"]
    optimizer_state_dict = model_data["optimizer_state_dict"]
    scheduler_state_dict = model_data["scheduler_state_dict"]
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler.load_state_dict(scheduler_state_dict)

    slim_train.global_steps = model_data["steps"]
    slim_train.best_loss = model_data["loss"]

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
                    slim_train.train(model, optimizer, scheduler, data,
                                     check_grad)

                    # update progress bars
                    minipb.set_postfix({"train loss": slim_train.train_loss})
                    slim_train.postfix["train loss"] = slim_train.train_loss
                    slim_train.trainpb.set_postfix(slim_train.postfix)

                    if check_grad:
                        vis.plot_grad_norm(slim_train.total_norm,
                                           slim_train.global_steps + 1,
                                           "average gradient norm")

                    # eval, each CHECK_POINT steps (every 5 epochs)
                    if slim_train.global_steps % (
                            CHECK_POINT -
                            1) == 0 or slim_train.global_steps == 0:
                        model_tested = True
                        slim_train.val_loss = 0
                        slim_train.val_steps = 0
                        for val_batch in val_iter:
                            val_mini_batches = get_mini_batch(
                                data=val_batch[0], size_=1)
                            slim_train.eval(model, val_mini_batches)

                        slim_train.val_loss = \
                            slim_train.val_loss / slim_train.val_steps

                        # update main progress bar
                        slim_train.postfix["test loss"] = slim_train.val_loss
                        slim_train.trainpb.set_postfix(slim_train.postfix)

                        # plot validation
                        vis.plot_loss(slim_train.val_loss,
                                      slim_train.global_steps + 1,
                                      "Validation")

                        # save model
                        val_loss = round(slim_train.val_loss, 2)
                        best_loss = round(slim_train.best_loss, 2)
                        if val_loss <= best_loss:
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
                minipb.close()
                train_pb.close()
                if not slim_train.in_train:
                    print("\nTraining finished ...")
                break

# %% [markdown]
#  ### Load the best model

# %%

# change device
flair.device = device
model_name = "slim_18-03-08h06_7485.pt"
load_path = model_path + model_name
model_data = torch.load(load_path, map_location=device)
model_state_dict = model_data["model_state_dict"]

model_inference = SLIM(model_parameters)
model_inference = model_inference.to(device)
model_inference.load_state_dict(model_state_dict)

# %% [markdown]
# ### Tesing Model

# %%
import pandas as pd

model_inference.eval()
test_loss = []
images_r = []
images_g = []
images_t = []
captions = []
view_image = []
views = []
test_steps = 0

with tqdm(test_iter, unit="file") as testpb:
    for test_batch in testpb:
        test_mini_batches = get_mini_batch(data=test_batch[0], size_=1)

        for mini_batch in test_mini_batches:
            testpb.set_description(f"Testing Step {test_steps}")
            with torch.no_grad():
                image, loss = model_inference(mini_batch)
                # generate image
                images_r.append(image)
                image_g = model_inference.generate(mini_batch)
                images_g.append(image_g.cpu())
                images_t.append(mini_batch[0])
                captions.append(mini_batch[3])
                view_image.append(mini_batch[1])
                views.append(mini_batch[2])

                test_loss.append(loss.item())
                test_steps += 1
data_dict = {
    "GroundTruth": images_t,
    "ImageGen": images_g,
    "ImageRecon": images_r,
    "ImageView": view_image,
    "SceneViews": views,
    "Captions": captions,
    "loss": test_loss
}

test_df = pd.DataFrame(data_dict)
test_df.index.name = "serial"

# %% [markdown]
# ### Loss Calculation

# %%
test_loss = np.array(test_loss)
loss_average = np.mean(test_loss)
loss_res = np.std(test_loss)
max_loss = np.max(test_loss)
min_loss = np.min(test_loss)

print(f"minimum loss: {min_loss:.3f}")
print(f"maximum loss: {max_loss:.3f}")
print(f"loss:         {loss_average:.2f} \u00B1 {loss_res:0.2f}")

# %%
test_loss = []
images_ = []
images_g = []
images_t = []
captions = []
test_steps = 0

with tqdm(test_iter, unit="file") as testpb:
    for test_batch in testpb:
        test_mini_batches = get_mini_batch(data=test_batch[0], size_=1)

        for mini_batch in test_mini_batches:
            testpb.set_description(f"Testing Step {test_steps}")
            with torch.no_grad():
                j = random.randint(0, 8)
                views_other = mini_batch[2][:, j, :].view(1, 1, -1)
                captions_text = mini_batch[3][j:j + 1]
                mybatch = [
                    mini_batch[0], mini_batch[1], views_other, captions_text
                ]
                image, loss = model_inference(mybatch)
                # generate image
                images_.append(image)
                image_g = model_inference.generate(mini_batch)
                images_g.append(image_g.cpu())
                images_t.append(mini_batch[0])
                captions.append(mini_batch[3])

                test_loss.append(loss.item())
                test_steps += 1

# %% [markdown]
#

# %%
k = 15
idx = np.argpartition(test_loss, k)
print(test_loss[idx][:6])

# %%
import matplotlib.pyplot as plt
plt.imshow(torch.squeeze(images_g[idx[5]]).cpu().permute(1, 2, 0))

# %%
plt.imshow(torch.squeeze(images_t[idx[5]]).cpu().permute(1, 2, 0))

# %%
plt.imshow(torch.squeeze(images_[idx[5]]).cpu().permute(1, 2, 0))

# %%
mini_batch[3][1:3].shape

# %%
