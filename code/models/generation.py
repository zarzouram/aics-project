"""
This code is edited from
Ref:
https://github.com/wohlert/generative-query-network-pytorch/tree/32f32632462d6796ecbc749a6617d3f3c0571b80

"""

import torch
# from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch.distributions as Dist

from layers.conv_lstm_conv import ConvLSTMCell
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

        self.z_size = z_size
        self.h_size = h_size
        self.T = iter_num

        # Recurrent encoder/decoder models
        self.encoder = ConvLSTMCell(2 * img_c,
                                    h_size,
                                    kernel_size=5,
                                    stride=2,
                                    padding=2)

        self.decoder = ConvLSTMCell(z_size + img_c + cond_size,
                                    h_size,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)

        # image read and imnage write
        self.write = nn.Conv2d(h_size,
                               img_c * 4,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.read = nn.Conv2d(img_c, img_c, kernel_size=3, stride=2, padding=1)

        # Outputs parameters of distributions
        self.variational = nn.Conv2d(h_size,
                                     2 * z_size,
                                     kernel_size=5,
                                     stride=1,
                                     padding=2)

        self.prior = nn.Conv2d(h_size,
                               2 * z_size,
                               kernel_size=5,
                               stride=1,
                               padding=2)

        self.loss = nn.MSELoss(reduction="none")

    def forward(self, x, cond):
        batch_size = x.size(0)
        device = self.model_param.device
        # Hidden states initialization
        (h_enc, c_enc) = self.encoder.init_hidden(batch_size=batch_size,
                                                  hidden=self.h_size,
                                                  shape=self.x_dims)

        (h_dec, c_dec) = self.decoder.init_hidden(batch_size=batch_size,
                                                  hidden=self.h_size,
                                                  shape=self.x_dims)

        canvas = x.new_zeros((batch_size, self.c, self.h, self.w),
                             device=device,
                             requires_grad=True)

        kl = 0
        for _ in range(self.T):
            # Reconstruction error
            epsilon = x - canvas

            # Infer posterior density from hidden state
            h_enc, c_enc = self.encoder(torch.cat([x, epsilon], dim=1), h_enc,
                                        c_enc)

            # Prior distribution
            p_mu, p_log_std = torch.split(self.prior(h_dec),
                                          self.z_size,
                                          dim=1)
            p_std = torch.exp(p_log_std)
            p_prior = Dist.Normal(p_mu, p_std)

            # Posterior distribution
            q_mu, q_log_std = torch.split(self.variational(h_enc),
                                          self.z_size,
                                          dim=1)
            q_std = torch.exp(q_log_std)
            q_posterior = Dist.Normal(q_mu, q_std)

            # Sample from posterior
            # z = q_mu + (0.5 * q_log_std).exp() * torch.randn_like(q_log_std)
            # reparameterization trick, same as above
            z = q_posterior.rsample()

            canvas_next = self.read(canvas)

            # Send representation through decoder
            cond_ = cond.clone().view(batch_size, -1, 1, 1)
            cond_ = cond_.contiguous().repeat(1, 1, self.h // 2, self.w // 2)
            h_dec, c_dec = self.decoder(
                torch.cat([z, canvas_next, cond_], dim=1), h_dec, c_dec)

            # Refine representation
            canvas = canvas + F.pixel_shuffle(self.write(h_dec), 2)

            kl += Dist.kl.kl_divergence(q_posterior, p_prior)

        # Return the reconstruction and kl
        constxn_loss = torch.sum(self.loss(canvas, x).view(batch_size, -1),
                                 dim=1)
        kl_loss = torch.sum(kl.view(batch_size, -1), dim=1)
        loss = torch.mean(constxn_loss + kl_loss)

        return canvas, loss

    def generate(self, x, cond):
        """
        Sample from the prior to generate a new
        datapoint.
        :param x: tensor representing shape of sample
        """
        batch_size = x.size(0)

        h_dec = x.new_zeros(
            (batch_size, self.h_size, self.h // 2, self.w // 2))
        c_dec = x.new_zeros(
            (batch_size, self.h_size, self.h // 2, self.w // 2))

        canvas = x.new_zeros((batch_size, self.c, self.h, self.w))
        device = self.model_param.device

        for _ in range(self.T):
            z = torch.randn(batch_size,
                            self.c,
                            self.h // 2,
                            self.w // 2,
                            device=device)

            canvas_next = self.read(canvas)
            cond_ = cond.clone().view(batch_size, -1, 1, 1)
            cond_ = cond_.contiguous().repeat(1, 1, self.h // 2, self.w // 2)

            h_dec, c_dec = self.decoder(
                torch.cat([z, canvas_next, cond_], dim=1), h_dec, c_dec)

            canvas = canvas + F.pixel_shuffle(self.write(h_dec), 2)

        return canvas
