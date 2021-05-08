"""
This code is edited from
Ref:
https://github.com/wohlert/generative-query-network-pytorch/tree/32f32632462d6796ecbc749a6617d3f3c0571b80

"""
import torch
# from torch import Tensor
from torch import nn
# import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
# from torch.distributions.bernoulli import Bernoulli

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
        self.scale = 8

        self.z_size = z_size
        self.h_size = h_size
        self.e_size = h_size
        self.T = iter_num

        # Recurrent encoder/decoder models
        self.encoder = ConvLSTMCell(img_w // self.scale,
                                    2 * img_c,
                                    h_size,
                                    kernel_size=19,
                                    stride=7,
                                    padding=2)

        self.decoder = ConvLSTMCell(img_w // self.scale,
                                    z_size + img_c + cond_size,
                                    h_size,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)

        # read and write
        self.write = nn.Sequential(
            nn.Conv2d(h_size, h_size // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size // 2,
                               12,
                               kernel_size=6,
                               stride=4,
                               padding=1), nn.ReLU(),
            nn.ConvTranspose2d(12,
                               2 * img_c,
                               kernel_size=4,
                               stride=2,
                               padding=1), nn.ReLU())

        self.read = nn.Conv2d(img_c,
                              img_c,
                              kernel_size=17,
                              stride=7,
                              padding=1)

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

    def forward(self, x, cond):
        batch_size = x.size(0)
        device = self.model_param.device

        # Hidden states initialization
        shape = (self.h // self.scale, self.w // self.scale)
        (h_enc, c_enc) = self.encoder.init_hidden(batch_size=batch_size,
                                                  hidden=self.h_size,
                                                  shape=shape)

        (h_dec, c_dec) = self.decoder.init_hidden(batch_size=batch_size,
                                                  hidden=self.h_size,
                                                  shape=shape)

        r = x.new_zeros((batch_size, self.c, self.h, self.w), device=device)

        noise = Uniform(-1 / 265, 1 / 265).rsample(x.size()).to(x.device)
        x += noise

        kl = 0
        for _ in range(self.T):
            # Reconstruction error
            epsilon = x - r  # (B, 3, 64, 64)

            # Infer posterior density from hidden state (W//8)
            # (B, 128, 8, 8)
            h_enc, c_enc = self.encoder(torch.cat([x, epsilon], dim=1), h_enc,
                                        c_enc)

            # Posterior distribution
            # (B, 3, 8, 8): z channel = 3
            q_mu, q_log_var = torch.split(self.variational(h_enc),
                                          self.z_size,
                                          dim=1)
            q_std = torch.exp(q_log_var / 2)
            q_posterior = Normal(q_mu, q_std)

            # Sample from posterior
            # (B, 3, 8, 8)
            z = q_posterior.rsample()

            # Prior distribution
            # (B, 3, 8, 8): z channel = 3
            p_mu, p_log_std = torch.split(self.prior(h_dec),
                                          self.z_size,
                                          dim=1)
            p_std = torch.exp(p_log_std)
            p_prior = Normal(p_mu, p_std)

            # Send representation through decoder
            # (B, 3, 64, 64) ==> (B, 3, 8, 8)
            r_next = self.read(r)

            # (B, 128, 8, 8)
            cond_ = cond.clone().view(batch_size, -1, 1, 1)
            cond_ = cond_.contiguous().repeat(1, 1, self.h // 8, self.w // 8)
            h_dec, c_dec = self.decoder(torch.cat([z, r_next, cond_], dim=1),
                                        h_dec, c_dec)

            # write representation
            # (B, 128, 8, 8) ==> 2*(B, 3, 64, 64)
            r_mu, r_log_var = torch.split(self.write(h_dec), self.c, dim=1)
            r = r + r_mu

            # KL divergence
            log_qzx = q_posterior.log_prob(z)
            log_pz = p_prior.log_prob(z)
            kl += log_qzx - log_pz

        # Return the reconstruction and kl
        r_var = torch.exp(r_log_var)
        image_dist = Normal(r, r_var)
        constxn_loss = image_dist.log_prob(x)
        constxn_loss = torch.sum(constxn_loss.view(batch_size, -1), dim=1)
        kl_loss = torch.sum(kl.view(batch_size, -1), dim=1)
        loss = torch.mean(kl_loss - constxn_loss)

        return r, loss

    def generate(self, x, cond):
        """
        Sample from the prior to generate a new
        datapoint.
        :param x: tensor representing shape of sample
        """
        batch_size = x.size(0)

        h_dec = x.new_zeros(
            (batch_size, self.h_size, self.h // 8, self.w // 8))
        c_dec = x.new_zeros(
            (batch_size, self.h_size, self.h // 8, self.w // 8))

        r = x.new_zeros((batch_size, self.c, self.h, self.w))

        for _ in range(self.T):
            # p_mu, q_log_var = torch.split(self.prior(h_dec),
            #                               self.z_size,
            #                               dim=1)
            # p_std = torch.exp(q_log_var / 2)
            # z = Normal(p_mu, p_std).sample()
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

            r_mu, _ = torch.split(self.write(h_dec), self.c, dim=1)
            r = r + r_mu

        return r
