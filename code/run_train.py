import argparse
import json
from tqdm import tqdm

from typing import List, Tuple

import random

import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np

from dataset.dataset_batcher import SlimDataset
from models.SLIM import SLIM
from dataset.preprocessing import get_mini_batch
from helpers.train_helper import Trainer
from helpers.early_stopping import EarlyStopping
from helpers.scheduler import LinearDecayLR

from utils.gpu_cuda_helper import get_gpus_avail
from utils.visualization import Visualizations

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
CUDA_LAUNCH_BLOCKING = 1


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training SLIM Model")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/scratch/guszarzmo/aicsproject/data/slim/turk_data_torch/",
        help="SLIM Dataset directory.")

    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/guszarzmo@GU.GU.SE/LT2318/aics-project/code/config.json",
        help="path to config file.")

    parser.add_argument(
        "--checkpoint_model",
        type=str,
        default="",
        help="If you want to resume trainng, pass model name to resume from.")

    parser.add_argument(
        "--plot_env_name",
        type=str,
        default="loss_plot",
        help="Visdom env. name to plot the training and validation loss.")

    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="GPU device to be used")

    parser.add_argument("--check_grad",
                        type=str,
                        default="no",
                        help="GPU device to be used")

    args = parser.parse_args()

    return parser, args


def select_device(gpu_id: int) -> torch.device:
    # get gpus that have >=75% free space
    cuda_idx = get_gpus_avail()  # type: List[Tuple[int, float]]
    cuda_id = None
    if cuda_idx:
        if gpu_id != -1:
            selected_gpu_avail = next(
                (i for i, v in enumerate(cuda_idx) if v[0] == gpu_id), None)
            if selected_gpu_avail is not None:
                cuda_id = gpu_id  # selected gpu has suitable free space
        else:
            cuda_id = cuda_idx[0][0]  # gpu with the most avail free space

    if cuda_id is None:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cuda_id}")

    print(f"\ndevice selected: {device}")

    return device


def load_config_file(config_path: str) -> List[dict]:
    with open(config_path, "r") as f:
        configs = json.load(f)

    return configs


def load_model(model_parameters: dict,
               device: torch.device,
               scheduler_param: dict,
               checkpoint_path: str = ""):

    lr_init = scheduler_param["lr_init"]
    model = SLIM(model_parameters)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=0.0001)
    scheduler = LinearDecayLR(optimizer)
    model_data = None

    if checkpoint_path != "":
        model_data = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))
        model_state_dict = model_data["model_state_dict"]
        optimizer_state_dict = model_data["optimizer_state_dict"]
        scheduler_state_dict = model_data["scheduler_state_dict"]
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

    return model, optimizer, scheduler, model_data


def run_train(train,
              train_iter,
              val_iter,
              model,
              optimizer,
              scheduler,
              es,
              vis,
              mini_batch_size,
              checkpoint_interv,
              check_grad=False):
    while train.in_train:
        train.local_steps = 0  # 1 epoch = SAMPLE_NUM local steps
        train.train_loss = 0
        train.epoch_loss = 0
        # train
        train_pb = tqdm(total=train.epoch_intv, leave=False, unit="local_step")
        # load one file (max 64 samples per file)
        for train_batch in train_iter:
            # progress bar one step
            trn_mini_b = get_mini_batch(data=train_batch,
                                        size_=mini_batch_size)

            # train min batches
            for data in trn_mini_b:
                train.step(model, optimizer, scheduler, data)

                best_model = False
                # eval, each CHECK_POINT steps (every 5 epochs)
                if (train.global_steps + 1) % checkpoint_interv == 0:
                    train.val_loss = 0
                    train.val_steps = 0
                    for val_batch in val_iter:
                        val_mini_batches = get_mini_batch(data=val_batch,
                                                          size_=1)
                        train.eval(model, val_mini_batches)

                    train.val_loss = \
                        train.val_loss / train.val_steps

                    # update main progress bar
                    train.postfix["test loss"] = train.val_loss
                    train.trainpb.set_postfix(train.postfix)

                    # save model
                    val_loss = round(train.val_loss, 2)
                    best_loss = round(train.best_loss, 2)
                    if val_loss <= best_loss:
                        train.best_loss = train.val_loss
                        best_model = True

                    # plot validation, save plot
                    vis.plot_loss(train.val_loss, train.epoch, "Validation")
                    vis.vis.save([vis.env_name])

                    # early stopping
                    if es.step(train.val_loss):
                        train.train_loss = \
                            train.train_loss / train.local_steps
                        train.in_train = False

                # End of epoch: Reach number of samples
                if (train.global_steps + 1) % train.epoch_intv == 0:
                    train.epoch_finished = True
                    train.epoch_loss = train.epoch_loss / (train.local_steps +
                                                           1)
                    # plot, save plot
                    vis.plot_loss(train.epoch_loss, train.epoch, "Train")
                    vis.vis.save([vis.env_name])
                    train.postfix["epoch loss"] = train.epoch_loss
                    train.epoch += 1

                # Reach the end of train loop
                if (train.global_steps + 1) == train.end:
                    train.in_train = False
                    # self.train_loss = self.train_loss / self.local_steps

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

                train_pb.set_postfix({"train loss": train.train_loss})
                train.trainpb.set_postfix(train.postfix)

                train_pb.set_description(desc_minib)
                train.trainpb.set_description(decc_epoch1 + decc_epoch2)

                train.trainpb.update(1)
                train_pb.update(1)

                if train.epoch_finished or not train.in_train:
                    break

            if train.epoch_finished or not train.in_train:
                train.epoch_finished = False
                train_pb.close()
                if not train.in_train:
                    print("\nTraining finished ...")
                    train.trainpb.close()
                break


if __name__ == "__main__":

    # parse argument command
    parser, args = parse_arguments()

    # select a device
    device = select_device(args.gpu)

    # Load configuration file
    configs = load_config_file(args.config_path)

    # Run Visualization
    vis = Visualizations(env_name=args.plot_env_name)

    # load training and validation dataloader
    train_dataset = SlimDataset(root_dir=args.dataset_dir + "train")
    train_iter = DataLoader(
        train_dataset,
        batch_size=configs["train_dataloader"]["file_batch"],
        shuffle=True,
        num_workers=configs["train_dataloader"]["num_workers"],
        pin_memory=device.type == "cuda")

    val_dataset = SlimDataset(root_dir=args.dataset_dir + "valid")
    val_iter = DataLoader(
        val_dataset,
        batch_size=configs["val_dataloader"]["file_batch"],
        shuffle=True,
        num_workers=configs["train_dataloader"]["num_workers"],
        pin_memory=device.type == "cuda")

    # load model
    if args.checkpoint_model == "":
        checkpoint_path = ""
    else:  # resume from checkpoint
        checkpoint_path = configs["checkpoints_dir"] + args.checkpoint_model

    hyperparameters = configs["model_hyperparameter"]
    scheduler_parm = configs["scheduler_parm"]
    model, optimizer, scheduler, model_data = load_model(
        hyperparameters, device, scheduler_parm, checkpoint_path)

    # load trianer class
    train_param = configs["train_param"]
    # number of steps per epoch
    epoch_interval = int(train_param["samples_num"] /
                         train_param["mini_batch_size"])
    checkpoint_interv = train_param["checkpoint_interv"] * epoch_interval
    trainer = Trainer(model,
                      device,
                      epoch_interval=epoch_interval,
                      save_path=configs["checkpoints_dir"])
    if args.checkpoint_model != "":  # resume from checkpoint
        trainer.global_steps = model_data["steps"] + 1
        trainer.best_loss = model_data["loss"]
        trainer.epoch = model_data["epoch"]

    # early stop
    early_stop_patience = configs["early_stop_patience"]
    es = EarlyStopping(patience=early_stop_patience, min_delta=0.1)

    run_train(trainer, train_iter, val_iter, model, optimizer, scheduler, es,
              vis, train_param["mini_batch_size"], checkpoint_interv)
