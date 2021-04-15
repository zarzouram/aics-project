# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim

import flair

# import matplotlib.pyplot as plt

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

seed = 42
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
#  ### Initialize some dirty variables
#
# %%
resume = True
change_saved_plot = False
model_name = "reg_save_slim.pt"  # saved checkpoint

# %% [markdown]
#  ### Auotomatic GPU selection

# %%
# Select cuda device based on the free memory
# Most of the time some cudas are busy but available
# torch.cuda.empty_cache()
prefered_device = None

if prefered_device is None:
    from utils.gpu_cuda_helper import get_gpus_avail

    cuda_idx = get_gpus_avail()
    device = None
    if not cuda_idx:
        device = torch.device("cpu")
    elif len(cuda_idx) >= 1:
        cuda_id = cuda_idx[0][0]
        # if cuda_id == 0:
        #     cuda_id = cuda_idx[1][0]
        device = torch.device(f"cuda:{cuda_id}")
else:
    device = torch.device(prefered_device)

# if use_cpu:
#     device = torch.device("cpu")

print(f"\ndevice selected: {device}")
flair.device = device

# %% [markdown]
#  ### Initialize Loss Visulaization

# %%
from utils.visualization import Visualizations
import json
from pathlib import Path

home = str(Path.home())
# Visualization
env_name = "Loss_Plot_1"
if resume and change_saved_plot:
    saved_step = int(model_name.split(".")[0].split("_")[-1])
    with open(f"{home}/.visdom/{env_name}.json") as f:
        data = json.load(f)
    for i in [0, 1]:
        x = data["jsons"][f"{env_name}_win"]["content"]["data"][i]["x"]
        y = data["jsons"][f"{env_name}_win"]["content"]["data"][i]["y"]
        idx = x.index(saved_step + 1) + 1
        x = x[:idx]
        y = y[:idx]
        data["jsons"][f"{env_name}_win"]["content"]["data"][i]["x"] = x
        data["jsons"][f"{env_name}_win"]["content"]["data"][i]["y"] = y

    with open(f"{home}/.visdom/{env_name}.json", "w") as json_file:
        json.dump(data, json_file)

vis = Visualizations(env_name)

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

# %%
# import numpy as np

# tokens_num = []
# for batch_train in train_iter:
#     for data in batch_train:
#         for sent in data[2][0]:
#             tokens_num.append(len(sent))

# plt.hist(tokens_num, bins="auto")

# %% [markdown]
#  ### Construct the model

# %%
model = SLIM(model_parameters)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr_init)
scheduler = LinearDecayLR(optimizer)

if resume:
    load_path = model_path + model_name
    model_data = torch.load(load_path, map_location=torch.device("cpu"))
    model_state_dict = model_data["model_state_dict"]
    optimizer_state_dict = model_data["optimizer_state_dict"]
    scheduler_state_dict = model_data["scheduler_state_dict"]
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler.load_state_dict(scheduler_state_dict)

# %% [markdown]
#  ### Model Train

# %%

print()
train = Trainer(model,
                device,
                epoch_interval=EPOCH_STEP,
                save_path=model_path,
                check_model_size=False)

# print(f"Model memory Size: {train.model_memory_gb:.2f} GB")

# %%
# early stopping
es = EarlyStopping(patience=10, min_delta=0.1)

# %%
check_grad = False

if resume:
    train.global_steps = model_data["steps"] + 1
    train.best_loss = model_data["loss"]
    train.epoch = model_data["epoch"]

while train.in_train:
    train.local_steps = 0  # 1 epoch = SAMPLE_NUM local steps
    train.train_loss = 0

    # train
    train_pb = tqdm(total=EPOCH_STEP, leave=False, unit="file")
    # load one file (max 64 samples per file)
    for train_batch in train_iter:
        # progress bar one step
        trn_mini_b = get_mini_batch(data=train_batch[0], size_=MINI_BATCH_SZ)

        with tqdm(trn_mini_b, leave=False, unit="minibatch") as minipb:
            minipb.set_description("minibatch train")

            # train min batches
            for data in minipb:
                train.step(model, optimizer, scheduler, data, check_grad)

                best_model = False
                # eval, each CHECK_POINT steps (every 5 epochs)
                if (train.global_steps + 1) % CHECK_POINT == 0:
                    train.val_loss = 0
                    train.val_steps = 0
                    for val_batch in val_iter:
                        val_mini_batches = get_mini_batch(data=val_batch[0],
                                                          size_=1)
                        train.eval(model, val_mini_batches)

                    train.val_loss = \
                        train.val_loss / train.val_steps

                    # update main progress bar
                    train.postfix["test loss"] = train.val_loss
                    train.trainpb.set_postfix(train.postfix)

                    # plot validation, save plot
                    vis.plot_loss(train.val_loss, train.global_steps + 1,
                                  "Validation")
                    vis.vis.save([vis.env_name])

                    # save model
                    val_loss = round(train.val_loss, 2)
                    best_loss = round(train.best_loss, 2)
                    if val_loss <= best_loss:
                        train.best_loss = train.val_loss
                        best_model = True

                    # early stopping
                    if es.step(train.val_loss):
                        train.train_loss = \
                            train.train_loss / train.local_steps
                        train.in_train = False

                # End of epoch: Reach number of samples
                if (train.global_steps + 1) % train.epoch_intv == 0:
                    train.epoch += 1
                    train.epoch_finished = True
                    # self.train_loss = self.train_loss / self.local_steps

                # Reach the end of train loop
                if (train.global_steps + 1) == train.end:
                    train.in_train = False
                    # self.train_loss = self.train_loss / self.local_steps

                # plot, save plot
                vis.plot_loss(train.train_loss, train.global_steps + 1,
                              "Train")

                if check_grad:
                    vis.plot_grad_norm(train.total_norm,
                                       train.global_steps + 1,
                                       "average gradient norm")

                if train.epoch_finished:
                    # save model and plot
                    train.save_checkpoint(model,
                                          optimizer,
                                          scheduler,
                                          best_model=best_model)
                    vis.vis.save([vis.env_name])

                train.local_steps += 1
                train.global_steps += 1

                # update progress bars
                desc_minib = f"LocalStep {train.local_steps}"
                decc_epoch1 = f"Global Step {train.global_steps} "
                decc_epoch2 = f"- epoch: {train.epoch}"
                train.postfix["train loss"] = train.train_loss

                minipb.set_postfix({"train loss": train.train_loss})
                train_pb.set_postfix(train.postfix)
                train.trainpb.set_postfix(train.postfix)

                train_pb.set_description(desc_minib)
                train.trainpb.set_description(decc_epoch1 + decc_epoch2)

                train.trainpb.update(1)
                train_pb.update(1)

                if train.epoch_finished or not train.in_train:
                    break

        if train.epoch_finished or not train.in_train:
            train.epoch_finished = False
            minipb.close()
            train_pb.close()
            if not train.in_train:
                print("\nTraining finished ...")
                train.trainpb.close()
            break

# %% [markdown]
#  ### Load the best model

# %%

# change device
# torch.device("cuda:0")
# flair.device = device
# model_name = "slim_01-04-14h36_4990.pt"
# load_path = model_path + model_name
# model_data = torch.load(load_path, map_location=device)
# model_state_dict = model_data["model_state_dict"]

# model_inference = SLIM(model_parameters)
# model_inference = model_inference.to(device)
# model_inference.load_state_dict(model_state_dict)

# # %% [markdown]
# # ### Tesing Model

# # %%
# import pandas as pd

# model_inference.eval()
# test_loss = []
# images_r = []
# images_g = []
# images_t = []
# captions = []
# view_image = []
# views = []
# test_steps = 0

# with tqdm(test_iter, unit="file") as testpb:
#     for test_batch in testpb:
#         test_mini_batches = get_mini_batch(
#             data=test_batch[0], size_=1)

#         for mini_batch in test_mini_batches:
#             testpb.set_description(f"Testing Step {test_steps}")
#             with torch.no_grad():
#                 image, loss = model_inference(mini_batch)
#                 # generate image
#                 images_r.append(image)
#                 image_g = model_inference.generate(mini_batch)
#                 images_g.append(image_g.cpu())
#                 images_t.append(mini_batch[0])
#                 captions.append(mini_batch[3])
#                 view_image.append(mini_batch[1])
#                 views.append(mini_batch[2])

#                 test_loss.append(loss.item())
#                 test_steps += 1
# data_dict = {"GroundTruth": images_t,
#              "ImageGen": images_g,
#              "ImageRecon": images_r,
#              "ImageView": view_image,
#              "SceneViews": views,
#              "Captions": captions,
#              "loss": test_loss}

# test_df =  pd.DataFrame(data_dict)
# test_df.index.name = "serial"

# # %% [markdown]
# # ### Loss Calculation

# # %%
# test_loss = np.array(test_loss)
# loss_average = np.mean(test_loss)
# loss_res = np.std(test_loss)
# max_loss = np.max(test_loss)
# min_loss = np.min(test_loss)

# print(f"minimum loss: {min_loss:.3f}")
# print(f"maximum loss: {max_loss:.3f}")
# print(f"loss:         {loss_average:.2f} \u00B1 {loss_res:0.2f}")

# # %% [markdown]
# # ### PLot Images

# # %%
# from torchvision.utils import make_grid

# def plot_grid(df):
#     plt.figure(figsize=(20, 12))

#     plt.subplot(311)
#     img = torch.squeeze(torch.stack(df.GroundTruth.to_list())).cpu()
#     grid = make_grid(img).permute(1, 2, 0).numpy()
#     plt.imshow(grid, interpolation="nearest")
#     plt.title("Ground Truth")

#     plt.subplot(312)
#     img = torch.squeeze(torch.stack(df.ImageRecon.to_list())).cpu()
#     grid = make_grid(img).permute(1, 2, 0).numpy()
#     plt.imshow(grid, interpolation="nearest")
#     plt.title("Reconstructed")

#     plt.subplot(313)
#     img = torch.squeeze(torch.stack(df.ImageGen.to_list())).cpu()
#     grid = make_grid(img).permute(1, 2, 0).numpy()
#     plt.imshow(grid, interpolation="nearest")
#     plt.title("Generated")

#     plt.tight_layout()
#     plt.show()

# # %%
# best_samples_df = test_df.nsmallest(5, 'loss', keep='all')
# plot_grid(best_samples_df)

# # %%
# worst_samples_df = test_df.nlargest(5, 'loss', keep='all')
# plot_grid(worst_samples_df)

# # %%

# # %% [markdown]
# # ### Change Views

# # %%
# j = random.randint(0,8)
# image_true = best_samples_df.GroundTruth.to_list()[4]
# image_vw = best_samples_df.ImageView.to_list()[4]
# views_other = best_samples_df.SceneViews.to_list()[4]
# captions = best_samples_df.Captions.to_list()[4]

# images_g = []
# images_r = []
# for j in tqdm(range(9)):
#     views_other_ = views_other[:, j, :].view(1, 1, -1)
#     captions_text = captions[j:j+1]
#     mybatch = [image_true, image_vw, views_other_, captions_text]
#     image, loss = model_inference(mybatch)
#     image_g = model_inference.generate(mini_batch)

#     images_r.append(image)
#     images_g.append(image_g)

# # %%
# plt.figure(figsize=(25, 15))

# plt.subplot(311)
# img = image_true.cpu()
# grid = make_grid(img).permute(1, 2, 0).numpy()
# plt.imshow(grid, interpolation="nearest")
# plt.title("Ground Truth")

# plt.subplot(312)
# img = torch.squeeze(torch.stack(images_r)).cpu()
# grid = make_grid(img, nrow=9).permute(1, 2, 0).detach().numpy()
# plt.imshow(grid, interpolation="nearest")
# plt.title("Reconstructed")

# plt.subplot(313)
# img = torch.squeeze(torch.stack(images_g)).cpu()
# grid = make_grid(img, nrow=9).permute(1, 2, 0).detach().numpy()
# plt.imshow(grid, interpolation="nearest")
# plt.title("Generated")

# %%
# model_inference.gen_model.encoder

# %%
