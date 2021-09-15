"""
This code is edited from
Ref:
https://github.com/wohlert/generative-query-network-pytorch/tree/32f32632462d6796ecbc749a6617d3f3c0571b80

"""
import math
import torch
# from torch import Tensor
from torch import nn
# import torch.nn.functional as F
from torch.distributions.normal import Normal
# from torch.distributions.bernoulli import Bernoulli

from layers.conv_lstm_simple import ConvLSTMCell
"""
The equation numbers on the comments corresponding
to the relevant equation given in the paper:
DRAW: A Recurrent Neural Network For Image Generation.
"""


class DRAW(nn.Module):
    def __init__(
        self,
        img_w: int,
        img_h: int,
        img_c: int,
        iter_num: int,
        h_size: int,
        z_size: int,
        cond_size: int,
    ):
        """
        Parameters
        ----------
        img_w:          int
                        image width
        img_h:          int
                        image heigth
        img_c:          int
                        image channel
        iter_num:       int
                        Number of time steps
        h_size:         int
        z_size:         int
                        latent space size
        cond_size:      int
                        The conditioning variable size (r + vwp)
        """
        super(DRAW, self).__init__()

        self.model_param = nn.Parameter(torch.empty(0))

        self.h = img_h
        self.w = img_w
        self.x_dims = (img_w, img_h)
        self.c = img_c
        self.scale = 8

        self.z_size = z_size
        self.h_size = h_size
        self.T = iter_num

        # Recurrent encoder/decoder models
        self.encoder = ConvLSTMCell("encode",
                                    img_w // self.scale,
                                    3 * img_c,
                                    h_size,
                                    kernel_size=19,
                                    stride=7,
                                    padding=2)

        self.decoder = ConvLSTMCell("decode",
                                    img_w,
                                    z_size * 2 + cond_size,
                                    img_c,
                                    kernel_size=19,
                                    stride=7,
                                    padding=2)

        # read and write
        self.write_mu = nn.Sequential(
            nn.Conv2d(img_c,
                      img_c,
                      kernel_size=3,
                      stride=1,
                      padding=2,
                      dilation=2), nn.GroupNorm(1, img_c))

        self.write_log_var = nn.Sequential(
            nn.Conv2d(img_c,
                      img_c,
                      kernel_size=3,
                      stride=1,
                      padding=2,
                      dilation=2), nn.GroupNorm(1, img_c), nn.ReLU())

        self.read = nn.Sequential(
            nn.Conv2d(img_c, z_size, kernel_size=17, stride=7, padding=1),
            nn.GroupNorm(1, z_size))

        # Outputs parameters of distributions
        self.variational = nn.Conv2d(h_size,
                                     2 * z_size,
                                     kernel_size=5,
                                     stride=1,
                                     padding=2)

    def forward(self, x, cond, var_scale):
        batch_size = x.size(0)
        device = self.model_param.device

        # Hidden states initialization
        # encoder hidden state size = (B, 128, 8, 8)
        # decoder hidden state size = (B, 3, 64, 64)
        shape = (self.h // self.scale, self.w // self.scale)
        (h_enc, c_enc) = self.encoder.init_hidden(batch_size=batch_size,
                                                  channel_size=self.h_size,
                                                  shape=shape)

        (h_dec, c_dec) = self.decoder.init_hidden(batch_size=batch_size,
                                                  channel_size=self.c,
                                                  shape=(self.h, self.w))

        r = x.new_zeros((batch_size, self.c, self.h, self.w), device=device)

        kl = 0
        for _ in range(self.T):
            # img_c = 3
            # z_size = 64
            # h_size = 128

            # Reconstruction error
            epsilon = x - r  # (B, 3, 64, 64)

            # Infer posterior density from hidden state (W//8)
            # (B, 3+3+3, 64, 64) ==> (B, 128, 8, 8)
            h_enc, c_enc = self.encoder(torch.cat([x, epsilon, h_dec], dim=1),
                                        h_enc, c_enc)

            # Posterior distribution
            # (B, 128, 8, 8) ==> 2*(B, 64, 8, 8)
            q_mu, q_log_var = torch.split(self.variational(h_enc),
                                          self.z_size,
                                          dim=1)
            q_std = torch.exp(q_log_var)
            q_posterior = Normal(q_mu, q_std)

            # Sample from posterior
            # (B, 64, 8, 8)
            z = q_posterior.rsample()

            # (B, 3, 64, 64) ==> (B, 64, 8, 8)
            r_next = self.read(r)

            # Send representation through decoder
            # (B, 64+64+96, 8, 8) ==> (B, 3, 64, 64)
            cond_ = cond.clone().view(batch_size, -1, 1, 1)
            cond_ = cond_.contiguous().repeat(1, 1, self.h // 8, self.w // 8)
            h_dec, c_dec = self.decoder(torch.cat([z, r_next, cond_], dim=1),
                                        h_dec, c_dec)

            # write representation
            # (B, 3, 64, 64) ==> (B, 3, 64, 64)
            r_mu = self.write_mu(h_dec)
            r = r + r_mu

            # KL divergence
            prior = Normal(torch.zeros_like(q_mu), torch.ones_like(q_std))
            log_qzx = q_posterior.log_prob(z)
            log_pz = prior.log_prob(z)
            kl += log_qzx - log_pz

        # (B, 3, 64, 64) ==> (B, 3, 64, 64)
        r_log_var = self.write_log_var(h_dec).view(batch_size, -1)

        # Return the reconstruction and kl
        # Gaussian negative log likelihood loss
        # source:
        # https://github.com/pytorch/pytorch/blob/6cdabb2e40a46a49ace66f5d94ed9c48bf6c3372/torch/nn/functional.py#L2597  # noqa:
        r_var = torch.exp(r_log_var)
        const = math.log(2 * math.pi)
        loss_ = ((x - r)**2).view(batch_size, -1)
        constxn_loss = 0.5 * (
            (r_log_var + loss_ / r_var).sum(dim=1) + const).mean()
        kl_loss = torch.sum(kl.view(batch_size, -1), dim=1).mean()
        loss = kl_loss + constxn_loss

        return r, loss, (kl_loss.item(), constxn_loss.item())

    def generate(self, x, cond):
        """
        Sample from the prior to generate a new
        datapoint.
        :param x: tensor representing shape of sample
        """
        batch_size = x.size(0)

        h_dec = x.new_zeros((batch_size, self.c, self.h, self.w))
        c_dec = x.new_zeros((batch_size, self.c, self.h, self.w))

        r = x.new_zeros((batch_size, self.c, self.h, self.w))

        for _ in range(self.T):
            z = torch.randn(batch_size,
                            self.z_size,
                            self.h // 8,
                            self.w // 8,
                            device=x.device)

            r_next = self.read(r)
            cond_ = cond.clone().view(batch_size, -1, 1, 1)
            cond_ = cond_.contiguous().repeat(1, 1, self.h // 8, self.w // 8)

            h_dec, c_dec = self.decoder(torch.cat([z, r_next, cond_], dim=1),
                                        h_dec, c_dec)

            r_ = self.write_mu(h_dec)
            r = r + r_

        return r
