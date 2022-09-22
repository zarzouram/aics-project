import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam

from codes.dataset.dataset_batcher import SlimDataset
from codes.models.SLIM import SLIM
from codes.helpers.scheduler import LinearDecayLR
from codes.helpers.train_helper import Trainer

from codes.utils.gpu_cuda_helper import select_device
from codes.utils.utils import seed_everything, save_config_file
from codes.utils.utils import load_config_file, seed_worker


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
        help="number of steps to freeze the DRAW module")

    parser.add_argument(
        '--device',
        type=str,
        default="gpu",  # gpu, cpu
        help='Device to be used {gpu, mgpu, cpu}')

    args = parser.parse_args()

    return parser, args


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
    vocab_path = None
    checkpoint_dir = Path(args.checkpoints_dir)
    if resume is None:
        time_tag = str(datetime.now().strftime("%d%m.%H%M"))  # exp. name
        checkpoint_dir = checkpoint_dir / time_tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_config_file(str(checkpoint_dir / "CONFIG_copy.json"), configs)

    else:  # resume from checkpoint
        time_tag = checkpoint_dir.parent
        checkpoint_path = checkpoint_dir / args.checkpoint_model
        vocab_path = checkpoint_dir / "vocab.pt"

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
    if freeze_gen != -1:
        freeze_gen *= configs_train["num_minibatch"]

    # seed
    seed = configs["seed"]
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # training and validation dataloader
    ds_dir = args.dataset_dir + "train"
    train_ds = SlimDataset(root_dir=ds_dir,
                           pretrain=pretrain,
                           glove_name=configs_glove["name"],
                           glove_dir=configs_glove["dir"],
                           glove_dim=configs_glove["dim"],
                           vocab_specials=vocab_specials,
                           vocab_path=vocab_path)
    collate_fn = None if train_ds.tokens is None else train_ds.collate_fn
    sampler = RandomSampler(train_ds, num_samples=num_samples)
    train_iter = DataLoader(train_ds,
                            collate_fn=collate_fn,
                            pin_memory=device.type == "cuda",
                            sampler=sampler,
                            worker_init_fn=seed_worker,
                            generator=g,
                            **config_loader["train"])
    if resume is None and pretrain != "draw":
        vocab_path = checkpoint_dir / "vocab.pt"
        torch.save(train_ds.vocab, vocab_path)

    ds_dir = args.dataset_dir + "val"
    val_ds = SlimDataset(root_dir=ds_dir,
                         pretrain=pretrain,
                         glove_name=configs_glove["name"],
                         glove_dir=configs_glove["dir"],
                         glove_dim=configs_glove["dim"],
                         vocab_specials=vocab_specials,
                         vocab_path=vocab_path)
    collate_fn = None if val_ds.tokens is None else val_ds.collate_fn
    val_iter = DataLoader(
        val_ds,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        **config_loader["val"],
        worker_init_fn=seed_worker,
        generator=g,
    )

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
    val_interv = configs_train["num_minibatch"] * configs_train["val_interv"]
    step_interv = configs_train["num_minibatch"]
    optm_group = []
    if pretrain is None or pretrain == "caption_encoder":
        # transformet optimizer
        xfmr_optm = Adam(model.caption_encoder.parameters(),
                         lr=1,
                         betas=config_optm["caption_encoder_beta"])
        xmfr_scheduler = LinearDecayLR(optimizer=xfmr_optm,
                                       lr_init=config_optm["xmfr_lr_init"],
                                       lr_final=config_optm["xmfr_lr_final"],
                                       step_num=configs_train["num_steps"],
                                       step_interv=step_interv)

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
        if freeze_gen == -1:
            optm_group.extend(draw_parms)
        optm = Adam(optm_group, lr=1)
        scheduler = LinearDecayLR(optimizer=optm,
                                  lr_init=config_optm["gen_lr_init"],
                                  lr_final=config_optm["gen_lr_final"],
                                  step_num=configs_train["num_steps"],
                                  step_interv=step_interv)

    if pretrain is None:
        optimizers = [xfmr_optm, optm]
        schedulers = [xmfr_scheduler, scheduler]
    elif pretrain == "draw":
        optimizers = [optm]
        schedulers = [scheduler]
    else:
        optimizers = [xfmr_optm]
        schedulers = [xmfr_scheduler]

    log_dir = f"logs/{time_tag}"
    train = Trainer(model=model,
                    optims=optimizers,
                    schedulers=schedulers,
                    train_iter=train_iter,
                    val_iter=val_iter,
                    device=device,
                    device_ids=device_ids,
                    val_interv=val_interv,
                    save_path=checkpoint_dir,
                    log_dir=log_dir,
                    seed=seed,
                    sigmas_const=configs_train["sigmas_const"],
                    total_steps=configs_train["num_steps"],
                    pretrain=pretrain,
                    freeze_gen=freeze_gen)
    if resume:
        train.resume(checkpoint_path)
    else:
        train.run()
