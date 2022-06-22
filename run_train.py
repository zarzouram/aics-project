import argparse
import json
from tqdm import tqdm

from typing import List, Tuple

import random

import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np

from codes.dataset.dataset_batcher import SlimDataset
from codes.models.SLIM import SLIM
from codes.dataset.preprocessing import get_mini_batch
from codes.helpers.train_helper import Trainer
from codes.helpers.scheduler import LinearDecayLR, VarAnnealer

from codes.utils.gpu_cuda_helper import select_device
from codes.utils.utils import seed_everything


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training SLIM Model")
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/srv/data/zarzouram/lt2318/slim/turk_torch/",
                        help="SLIM Dataset directory.")

    parser.add_argument("--config_path",
                        type=str,
                        default="config.json",
                        help="path to config file.")

    parser.add_argument("--checkpoints_dir",
                        type=str,
                        default="/srv/data/zarzouram/lt2318/checkpoints",
                        help="path to config file.")

    parser.add_argument(
        "--checkpoint_model",
        type=str,
        default="",
        help="If you want to resume trainng, pass model name to resume from.")

    parser.add_argument(
        "--pretrain",
        type=str,
        default="draw",  #
        help="pretraining a submodule, {draw, caption_encoder}")

    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="GPU device to be used")

    args = parser.parse_args()

    return parser, args


def load_config_file(config_path: str) -> List[dict]:
    with open(config_path, "r") as f:
        configs = json.load(f)

    return configs


def load_model(model_parameters: dict,
               scheduler_param: dict,
               checkpoint_path: str = ""):

    lr_init = scheduler_param["lr_init"]
    model = SLIM(model_parameters)
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    scheduler = LinearDecayLR(optimizer, **scheduler_param)
    model_data = None
    # Variance scales
    var_scale = VarAnnealer(**configs["var_scale_parm"])

    if checkpoint_path != "":
        model_data = torch.load(checkpoint_path,
                                map_location=torch.device("cpu"))
        model_state_dict = model_data["model_state_dict"]
        optimizer_state_dict = model_data["optimizer_state_dict"]
        scheduler_state_dict = model_data["scheduler_state_dict"]
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)
        var_scale.scale = model_data["var_scale"]
        var_scale.t = model_data["steps"] + 1

    return model, optimizer, scheduler, var_scale, model_data


def run_train(train,
              train_iter,
              val_iter,
              model,
              optimizer,
              scheduler,
              configs,
              var_scale,
              vis,
              win_name,
              check_grad=False):

    # other training param
    train_param = configs["train_param"]
    mini_batch_size = train_param["mini_batch_size"]
    checkpoint_interv = train_param["checkpoint_interv"] * train.epoch_intv

    while train.in_train:
        train.local_steps = 0  # 1 epoch = SAMPLE_NUM local steps
        train.train_loss = 0
        train.epoch_loss = 0
        train.kl_tain = 0
        train.lx_train = 0
        # train progress bar
        train_pb = tqdm(total=train.epoch_intv, leave=False, unit="local_step")
        # load one file (max 64 samples per file)
        for train_batch in train_iter:
            # progress bar one step
            trn_mini_b = get_mini_batch(data=train_batch,
                                        size_=mini_batch_size)

            # train min batches
            for data in trn_mini_b:

                vs = next(var_scale)
                train.step(model, optimizer, scheduler, data, vs)

                best_model = False
                # eval, each CHECK_POINT steps (every 5 epochs)
                if (train.global_steps + 1) % checkpoint_interv == 0:
                    train.val_loss = 0
                    train.kl_val = 0
                    train.lx_val = 0
                    train.val_steps = 0
                    for val_batch in val_iter:
                        val_mini_batches = get_mini_batch(data=val_batch,
                                                          size_=1)
                        train.eval(model, val_mini_batches, var_scale.scale)

                    train.val_loss = \
                        train.val_loss / train.val_steps
                    train.kl_val = \
                        train.kl_val / train.val_steps
                    train.lx_val = \
                        train.lx_val / train.val_steps

                    # update main progress bar
                    train.postfix["test loss"] = train.val_loss
                    train.trainpb.set_postfix(train.postfix)

                    # plot validation, save plot
                    vis.plot_line(train.val_loss, train.epoch, "Validation",
                                  win_name[0])
                    if len(vis.win_name.keys()) > 1:
                        vis.plot_line(train.kl_val, train.epoch, "Validation",
                                      win_name[2])
                        vis.plot_line(train.lx_val, train.epoch, "Validation",
                                      win_name[1])
                    vis.vis.save([vis.env_name])

                    # save model
                    val_loss = round(train.val_loss, 2)
                    best_loss = round(train.best_loss, 2)
                    if val_loss <= best_loss:
                        train.best_loss = train.val_loss
                        best_model = True

                    # # early stopping
                    # if es.step(train.val_loss):
                    #     train.train_loss = \
                    #         train.train_loss / train.local_steps
                    #     train.in_train = False

                # End of epoch: Reach number of samples
                if (train.global_steps + 1) % train.epoch_intv == 0:
                    train.epoch_finished = True
                    train.epoch_loss = train.epoch_loss / (train.local_steps +
                                                           1)
                    train.lx_train = train.lx_train / (train.local_steps + 1)
                    train.kl_train = train.kl_train / (train.local_steps + 1)
                    # plot, save plot
                    vis.plot_line(train.epoch_loss, train.epoch, "Train",
                                  win_name[0])
                    if len(vis.win_name.keys()) > 1:
                        vis.plot_line(train.kl_train, train.epoch, "Train",
                                      win_name[2])
                        vis.plot_line(train.lx_train, train.epoch, "Train",
                                      win_name[1])
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
                                          var_scale.scale,
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
    device = select_device(args.device)

    # Load configuration file
    configs = load_config_file(args.config_path)

    # seed
    seed = configs["seed"]
    seed_everything(seed)

    # training and validation dataloader
    pretrain = args.pretrain  # type: str
    ds_dir = args.dataset_dir + "train"
    train_ds = SlimDataset(root_dir=ds_dir, pretrain=pretrain)
    collate_fn = None if train_ds.tokens is None else train_ds.collate_fn
    train_iter = DataLoader(train_ds,
                            collate_fn=collate_fn,
                            pin_memory=device.type == "cuda",
                            **configs["dataloader"]["train"])

    ds_dir = args.dataset_dir + "valid"
    val_ds = SlimDataset(root_dir=ds_dir, pretrain=pretrain)
    val_iter = DataLoader(val_ds,
                          collate_fn=collate_fn,
                          pin_memory=device.type == "cuda",
                          **configs["dataloader"]["val"])

    # load model
    if args.checkpoint_model == "":
        checkpoint_path = ""
    else:  # resume from checkpoint
        checkpoint_path = configs["checkpoints_dir"] + args.checkpoint_model

    hyperparameters = configs["model_hyperparameter"]
    scheduler_parm = configs["scheduler_parm"]
    model, optimizer, scheduler, var_scale, model_data = load_model(
        hyperparameters,
        device,
        scheduler_parm,
    )

    # load trianer class
    train_param = configs["train_param"]
    # number of steps per epoch
    epoch_intrv = int(train_param["samples_num"] /
                      train_param["mini_batch_size"])
    trainer = Trainer(model,
                      device,
                      epoch_interval=epoch_intrv,
                      save_path=configs["checkpoints_dir"])
    if args.checkpoint_model != "":  # resume from checkpoint
        trainer.global_steps = model_data["steps"] + 1
        trainer.best_loss = model_data["loss"]
        trainer.epoch = model_data["epoch"]

    # Init Visualization
    env_name = args.plot_env_name
    vis = Visualizations(env_name=env_name)
    legend = [["Train", "Validation"]]
    title = [f"Loss Plot (mean every 1 epoch/{epoch_intrv} steps)"]
    xlabel = [f"Epoch ({epoch_intrv} steps)"]
    ylabel = ["ELBO Loss"]
    win_name = [f"{env_name}_total_Loss"]
    if args.plot_loss_comp.lower() == "y":
        for loss_type in ["Const_Loss", "KlD_Loss"]:
            legend.append(["Train", "Validation"])
            title.append(
                f"{loss_type} Plot (mean every 1 epoch/{epoch_intrv} steps)")
            xlabel.append(f"Epoch ({epoch_intrv} steps)")
            ylabel.append(f"{loss_type}")
            win_name.append(f"{env_name}_{loss_type}")
    opt_win = {
        "win_name": win_name,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "title": title,
        "legend": legend
    }
    vis.add_wins(**opt_win)

    run_train(trainer, train_iter, val_iter, model, optimizer, scheduler,
              configs, var_scale, vis, win_name)
