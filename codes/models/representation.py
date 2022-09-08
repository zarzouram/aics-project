import torch
from torch import nn
from torch import Tensor


class RepresentationNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_dim: int,
                 output_size: int) -> None:
        """
        Parameters
        ----------
        rep_input_size:     nt
                            Size of scene representation input
        rep_output_size:    int
                            Size of scene representation output (r)
        """

        super(RepresentationNetwork, self).__init__()

        self.rep_output_size = output_size
        self.scene_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_size),
        )

        self.init_weights()

    def init_weights(self):
        for p in self.scene_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, cpt_embs: Tensor, viewpoints: Tensor) -> Tensor:

        # Scene representation
        # (B, N=9, CE+VE)
        scene_vectors = torch.cat((cpt_embs, viewpoints), dim=-1)

        # Scene embedding  h (B, N=9, SE)
        h = self.scene_encoder(scene_vectors)

        return h
