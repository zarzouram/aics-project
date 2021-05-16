import torch
from torch import nn
from torch import Tensor


class RepresentationNetwork(nn.Module):
    def __init__(self, caption_embs_size: int, viewpoints_size: int,
                 viewpoints_embs_size: int, r_size: int) -> None:
        """
        Parameters
        ----------
        bert_dir:               string
                                Bert model absolute directory
        caption_embs_size:      int
                                Size of sentence encoding
        viewpoints_size:        int
                                Size of camera viwpoints representation
        viewpoints_embs_size:   int
                                Size of camera viwpoints encoding
        r_size:                 int
                                Size of scene representation (r)
        """

        super(RepresentationNetwork, self).__init__()

        self.viewpoint_encoder = nn.Linear(viewpoints_size,
                                           viewpoints_embs_size)
        rep_input_size = caption_embs_size + viewpoints_embs_size
        self.r_size = r_size
        self.scene_encoder = nn.Sequential(
            nn.Linear(rep_input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, r_size),
        )

    def forward(self, cpt_embs: Tensor, viewpoints: Tensor, n: int) -> Tensor:

        # Encode viewpoints: (B*N, VE)
        views_encoded = self.viewpoint_encoder(viewpoints)
        # Scene representation
        scene_vectors = torch.cat((cpt_embs, views_encoded), dim=-1)
        # Scene embedding  h
        h = self.scene_encoder(scene_vectors)
        # aggregation
        r = h.view(-1, n, self.r_size).mean(1)
        return r
