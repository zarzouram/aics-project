import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam

from codes.dataset.dataset_batcher import SlimDataset
from codes.models.SLIM import SLIM
from codes.dataset.preprocessing import get_mini_batch
from codes.helpers.scheduler import XfmrWarmupScheduler, LinearDecayLR
# from codes.helpers.train_helper import Trainer

from codes.utils.gpu_cuda_helper import select_device
from codes.utils.utils import seed_everything, save_config_file
from codes.utils.utils import load_config_file


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training SLIM Model")
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/srv/data/zarzouram/lt2318/slim/turk_torch/",
                        help="SLIM Dataset directory.")

    parser.add_argument("--config_path",
                        type=str,
                        default="codes/config.json",
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

    parser.add_argument("--load_pretrain",
                        type=str,
                        default="",
                        help="path to pretrained module to be load")

    parser.add_argument(
        "--pretrain",
        type=str,
        default="",  #
        help="pretraining a submodule, {draw, caption_encoder}")

    parser.add_argument(
        "--freeze_gen",
        type=int,
        default=-1,  #
        help="number of epochs to freeze the DRAW module")

    parser.add_argument(
        '--device',
        type=str,
        default="mgpu",  # gpu, cpu
        help='Device to be used {gpu, mgpu, cpu}')

    args = parser.parse_args()

    return parser, args


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

    vocab_specials = {"pad": "<pad>", "eos": "<eos>", "unk": "<unk>"}

    # parse argument command
    parser, args = parse_arguments()
    freeze_gen = args.freeze_gen

    # Load configuration file
    configs = load_config_file(args.config_path)
    config_loader = configs["dataloader"]
    configs_train = configs["train_param"]
    configs_glove = configs["glove"]
    config_optm = configs["optim_params"]
    hyperparameters = configs["model_hyperparameter"]

    # training status
    resume = args.checkpoint_model if args.checkpoint_model else None
    load_pretrain = args.load_pretrain if args.load_pretrain else None
    train_status_logic = not (resume is not None and load_pretrain is not None)
    train_status_messege = "Either loading a checkpoint or a pretrained model"
    assert train_status_logic, train_status_messege

    pretrain = args.pretrain if args.pretrain else None
    freeze_logic = not (freeze_gen != -1 and pretrain is not None)
    freeze_mssg = "Can't freeze DRAW submodule and pretrain at the same time."
    assert freeze_logic, freeze_mssg

    # experiment status
    checkpoint_dir = Path(args.checkpoints_dir)
    if resume is None:
        time_tag = str(datetime.now().strftime("%d%m.%H%M"))  # exp. name
        checkpoint_dir = checkpoint_dir / time_tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_config_file(str(checkpoint_dir / "CONFIG_copy.json"), configs)

    else:  # resume from checkpoint
        time_tag = checkpoint_dir.parent
        checkpoint_path = checkpoint_dir / args.checkpoint_model
        vocab_path = checkpoint_dir + "vocab.pt"

    # select a device
    device = select_device(args.device)
    if isinstance(device, list):
        device_ids = device
        device = torch.device(f"cuda:{device[0]}")
        cudas = f"cuda:{device_ids[0]} & cuda:{device_ids[1]}"
        print(f"selected devices are {cudas}.\n")
    else:
        print(f"selected device is {device}.\n")
        device_ids = None

    # some parameters
    minibatch_size = config_loader["train"]["batch_size"]
    num_samples = minibatch_size * configs_train["num_minibatch"]
    num_steps = configs_train["num_steps"]

    # seed
    seed = configs["seed"]
    seed_everything(seed)

    # training and validation dataloader
    ds_dir = args.dataset_dir + "train"
    train_ds = SlimDataset(root_dir=ds_dir,
                           pretrain=pretrain,
                           glove_name=configs_glove["name"],
                           glove_dir=configs_glove["dir"],
                           glove_dim=configs_glove["dim"],
                           vocab_specials=vocab_specials)
    collate_fn = None if train_ds.tokens is None else train_ds.collate_fn
    sampler = RandomSampler(train_ds, num_samples=num_samples)
    train_iter = DataLoader(train_ds,
                            collate_fn=collate_fn,
                            pin_memory=device.type == "cuda",
                            sampler=sampler,
                            **config_loader["train"])

    ds_dir = args.dataset_dir + "val"
    val_ds = SlimDataset(root_dir=ds_dir,
                         pretrain=pretrain,
                         glove_name=configs_glove["name"],
                         glove_dir=configs_glove["dir"],
                         glove_dim=configs_glove["dim"],
                         vocab_specials=vocab_specials)
    collate_fn = None if val_ds.tokens is None else val_ds.collate_fn
    val_iter = DataLoader(val_ds,
                          collate_fn=collate_fn,
                          pin_memory=device.type == "cuda",
                          **config_loader["val"])

    # Construt model
    if pretrain is None or pretrain == "caption_encoder":
        hyperparameters["vocab_size"] = len(train_ds.vocab)
    model = SLIM(params=hyperparameters, pretrain=pretrain)
    if load_pretrain is not None:
        model.load_pretrained(load_pretrain)
        if freeze_gen != -1:  # freeze DRAW sub-module
            model.freeze_draw()

    elif pretrain is None or pretrain == "caption_encoder":
        vectors = train_ds.get_glove(train_ds.pad_value)
        model.caption_encoder.embedding.from_pretrained(vectors, freeze=False)

    # Optimizer
    optm_group = []
    if pretrain is None or pretrain == "caption_encoder":
        # transformet optimizer
        xfmr_optm = Adam(model.caption_encoder.parameters(),
                         lr=config_optm["caption_encoder_lr"],
                         betas=config_optm["caption_encoder_beta"],
                         eps=1e-9)
        xmfr_scheduler = XfmrWarmupScheduler(optimizer=xfmr_optm)

        # representation scene submodule optimizer
        if pretrain is None:
            optm_group.append({"params": model.viewpoint_encoder.parameters()})
            optm_group.append({"params": model.rep_model.parameters()})

    if pretrain is None or pretrain == "draw":
        draw_parms = [{
            "params": model.target_viewpoint_encoder.parameters()
        }, {
            "params": model.gen_model.parameters()
        }]
        if freeze_gen != -1:
            optm_group.append(draw_parms)
        optm = Adam(optm_group, lr=config_optm["lr_init"])
        scheduler = LinearDecayLR(optimizer=optm)
