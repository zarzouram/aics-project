from typing import Dict

from collections import defaultdict
from statistics import mean

import numpy as np
import torch
from torch import Tensor
from torchvision.utils import make_grid


class TrackMetrics:

    def __init__(self) -> None:
        self.reset_running()
        self.metrics = self.init_metrics()
        self.image_tracking = defaultdict(list)
        self.em = "loss"  # metrics used in eval is loss

    def create_default_dict(self):
        metrics_dict = {
            "train": defaultdict(list, {}),
            "val": defaultdict(list, {})
        }

        return metrics_dict

    def reset_running(self):
        self.running = self.create_default_dict()

    def reset_image_tracking(self):
        self.image_tracking = defaultdict(list)

    def init_metrics(self):
        return self.create_default_dict()

    def add_running(self, metrics: Dict[str, float], phase: str) -> None:
        for name, value in metrics.items():
            self.running[phase][name].append(value)

    def add_images(self, images: Dict[str, Tensor]) -> None:
        for name, value in images.items():
            self.image_tracking[name].append(value)

    def update(self, phase: str):
        for name, value in self.running[phase].items():
            self.metrics[phase][name].append(mean(value))
        self.reset_running()

    def track_best_images(self) -> Tensor:
        # you must track best images before updat the tracking metrics
        loss = self.running["val"][self.em]
        best_ids = np.array(loss).argsort()[:10]
        img_gt = torch.vstack(self.image_tracking["img_gt"])[best_ids]
        img_gn = torch.vstack(self.image_tracking["img_gn"])[best_ids]
        imgs = torch.vstack([img_gt, img_gn])
        img_grid = make_grid(imgs, nrow=10)

        self.reset_image_tracking()

        return img_grid

    def states(self):
        return {
            "running": self.running,
            "metrics": self.metrics,
            "images": self.image_tracking
        }

    def load_states(self, states: dict):
        self.running = states["running"]
        self.metrics = states["metrics"]
        self.image_tracking = states["images"]

    def last_metric(self):
        return round(self.metrics["val"][self.em][-1], 6)
