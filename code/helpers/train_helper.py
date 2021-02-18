import tqdm

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from helpers.visualization import Visualizations
from helpers.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore",
                        message="Please also save or load the state ")


class Trainer():
    def __init__(self, model, device, check_point: int, sample_num: int,
                 opt_config: dict, save_path: str, datasets: str):

        # model initialization
        self.model = model.to(device)

        # optimizer initialize
        self.__decay_rate = opt_config["decay_rate"]
        self.__lr_init = opt_config["lr_init"]
        lr0 = self.lr_decay(0)
        self.optimizer = optim.Adam(model.parameters())
        self.scheduler = LambdaLR(
            self.optimizer, lr_lambda=lambda step: self.lr_decay(step) / lr0)

        # data loader
        self.train_iter = DataLoader(datasets[0],
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=0)

        self.val_iter = DataLoader(datasets[1],
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=0)

        # Other parameters
        self.plot_point = 25
        self.CHECK_POINT = check_point  # check bestloss every CHECK_POINT steps
        self.global_steps = 0  # number of steps counter
        self.SAMPLE_NUM = sample_num  # number of samples to train every epoch
        self.best_loss = -1  # best valid_tn loss calc_d every CHECK_POINT steps
        self.val_img = None  # constructed images for the best model
        self.save_path = save_path  # path to save best mpdel
        self.epoch = 0
        self.trainpb = tqdm.tqdm(total=2000000)
        self.postfix = {"train/loss": 0.0, "test/loss": 0.0}
        self.in_train = True
        self.end = 2000000

        # early stopping
        self.es = EarlyStopping(patience=3)

        # Visualization
        self.vis = Visualizations()

    def train_eval(self):
        local_steps = 0
        running_loss = 0
        plot_loss = 0
        for batch in self.train_iter:
            for mini_batch in batch:
                self.trainpb.set_description(
                    f"Global Step {self.global_steps}")

                # train model
                self.model.train()
                self.optimizer.zero_grad()
                # forward
                loss = self.model(mini_batch)
                # backward and update
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                # update progress bar
                self.postfix["train/loss"] = loss.item()
                self.trainpb.set_postfix(self.postfix)

                local_steps += 1

                # plot
                plot_loss += loss.item()
                if self.global_steps == 0:
                    self.vis.plot_loss(plot_loss, self.global_steps, "Train")

                elif self.global_steps % (self.plot_point - 1) == 0:
                    plot_loss = plot_loss / self.plot_point
                    self.vis.plot_loss(plot_loss, self.global_steps, "Train")
                    plot_loss = 0

                # check_point validation
                if self.global_steps % (self.CHECK_POINT - 1) == 0:
                    val_loss = self.__validate()
                    # check best loss value
                    if val_loss > self.best_loss:
                        self.best_loss = val_loss
                        self.__save_checkpoint()

                    if self.global_steps % (500 - 1) == 0:
                        if self.es.step(val_loss):
                            running_loss = running_loss / local_steps
                            return running_loss

                # Reach numple of samples
                if self.global_steps % (self.SAMPLE_NUM -
                                        1) == 0 and self.global_steps != 0:
                    self.epoch += 1
                    running_loss = running_loss / local_steps
                    return running_loss

                # Reach the end of train loop
                if self.global_steps == (self.end + 1):
                    self.in_train = False
                    running_loss = running_loss / local_steps
                    return running_loss

                self.global_steps += 1
                self.trainpb.update(1)

                # gc.collect()
                torch.cuda.empty_cache()

        running_loss = running_loss / local_steps

        return running_loss

    def __validate(self):
        # validate
        running_loss = 0
        steps = 0
        self.model.eval()
        with tqdm.tqdm(self.val_iter,
                       leave=False,
                       unit="Batch",
                       total=len(self.val_iter)) as self.valpb:
            for val_batch in self.valpb:
                self.valpb.set_description(f"Step {steps}")

                with torch.no_grad():
                    loss = self.model(val_batch)
                    running_loss += loss.item()
                # update progress bar
                steps += 1
                self.valpb.set_postfix({"test/loss": loss.item()})
                self.valpb.update(1)

        current_loss = running_loss / steps
        self.postfix["test/loss"] = current_loss
        self.trainpb.set_postfix(self.postfix)
        # plot
        self.vis.plot_loss(current_loss, self.global_steps, "Validation")

        return running_loss / steps

    def __save_checkpoint(self):
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        scheduler_state = self.scheduler.state_dict()

        state_dict = {
            "steps": self.global_steps,
            "loss": self.best_loss,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state
        }

        torch.save(state_dict, self.save_path)

    def lr_decay(self, global_step: int):
        lr = -self.__decay_rate * global_step + self.__lr_init
        return lr
