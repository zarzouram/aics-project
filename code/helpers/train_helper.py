import tqdm

import torch

from utils.pytorch_modelsize import SizeEstimator

import warnings
warnings.filterwarnings("ignore",
                        message="Please also save or load the state ")


class Trainer():
    def __init__(self,
                 model,
                 device,
                 epoch_interval: int,
                 save_path: str,
                 check_model_size: bool = False):

        if check_model_size:
            se = SizeEstimator(model)
            se.get_parameter_sizes()
            se.calc_param_bits()
            sz = se.param_bits
            self.model_memory_gb = (sz / 8) / (1024**3)
        else:
            self.model_memory_gb = None

        # Other parameters
        self.global_steps = 0  # number of steps counter
        self.local_steps = 0
        self.val_steps = 0

        # number of samples to train every (epoch)
        self.epoch_finished = False
        self.epoch = 0
        self.epoch_intv = epoch_interval

        # losses
        self.train_loss = 0
        self.val_loss = 0
        self.best_loss = -1  # best valid_tn loss calc_d every CHECK_POINT steps

        self.save_path = save_path  # path to save best mpdel

        self.in_train = True
        self.end = 2000000

        # progress bar data
        self.trainpb = tqdm.tqdm(total=self.end, unit="GlobalStep")
        self.postfix = {"train/loss": 0.0, "test/loss": 0.0}
        self.trainpb.set_description(f"Global Step {self.global_steps}")

    def train(self, model, optimizer, scheduler, batch):
        # train model
        model.train()
        optimizer.zero_grad()
        # forward
        loss = model(batch)
        # backward and update
        loss.backward()
        optimizer.step()
        scheduler.step()

        self.train_loss = loss.item()
        self.local_steps += 1
        # self.train_loss = self.train_loss / self.local_steps

        # End of epoch: Reach number of samples
        if (self.global_steps + 1) % self.epoch_intv == 0:
            self.global_steps += 1
            self.epoch += 1
            self.epoch_finished = True
            # self.train_loss = self.train_loss / self.local_steps
            return

        # Reach the end of train loop
        if self.global_steps == (self.end - 1):
            self.global_steps += 1
            self.in_train = False
            # self.train_loss = self.train_loss / self.local_steps

        self.global_steps += 1

        # update progress bars
        self.trainpb.set_description(
            f"Global Step {self.global_steps} - epoch: {self.epoch}")
        self.trainpb.update(1)

        # torch.cuda.empty_cache()

        return

    def eval(self, model, val_batch):
        # validate
        model.eval()
        with tqdm.tqdm(val_batch, leave=False, unit="valbatch") as valpb:
            for val_mini_batch in valpb:

                with torch.no_grad():
                    loss = model(val_mini_batch)
                    self.val_loss += loss.item()

                self.val_steps += 1
                # update progress bars
                valpb.set_postfix({"test/loss": loss.item()})
                valpb.update(1)

        return

    def save_checkpoint(self, model):
        model_state = model.state_dict()
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
