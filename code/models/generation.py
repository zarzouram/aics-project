"""
This code is edited from
Ref:
https://github.com/wohlert/generative-query-network-pytorch/tree/32f32632462d6796ecbc749a6617d3f3c0571b80

"""
import torch
from torch import Tensor

from torch import nn
from torch.distributions import Normal, kl_divergence

from layers.conv_lstm_simple import ConvLSTMCell


class DRAW(nn.Module):

    def __init__(self,
                 img_w: int,
                 img_h: int,
                 img_c: int,
                 iter_num: int,
                 h_dim: int,
                 z_dim: int,
                 cond_size: int,
                 init_tunning: bool = False):
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
        h_dim:          int
        z_dim:          int
                        latent space size
        cond_size:      int
                        The conditioning variable size (r + vwp)
        """
        super(DRAW, self).__init__()

        self.h = img_h
        self.w = img_w
        self.x_dims = (img_w, img_h)
        self.c = img_c

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.T = iter_num

        self.init_tunning = init_tunning

        # Recurrent encoder/decoder models
        self.encoder = ConvLSTMCell(img_c + h_dim,
                                    h_dim,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)

        self.decoder = ConvLSTMCell(img_c + z_dim + cond_size,
                                    h_dim,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2)

        #
        self.write = nn.Conv2d(h_dim, 2 * img_c, kernel_size=1, stride=1)

        # parameters of distributions
        self.posterior = nn.Conv2d(h_dim,
                                   2 * z_dim,
                                   kernel_size=5,
                                   stride=1,
                                   padding=2)
        self.prior = nn.Conv2d(h_dim,
                               2 * z_dim,
                               kernel_size=5,
                               stride=1,
                               padding=2)

        self.init_hidden()

    def init_hidden(self):

        shape = (self.h, self.w)
        self.h_enc = torch.zeros(1, self.h_size, *shape)
        self.c_enc = torch.zeros(1, self.h_size, *shape)
        self.h_dec = torch.zeros(1, self.c, self.h, self.w)
        self.c_dec = torch.zeros(1, self.c, self.h, self.w)

        if self.init_tunning:
            self.h_enc = nn.Parameter(self.h_enc, requires_grad=True)
            self.c_enc = nn.Parameter(self.c_enc, requires_grad=True)
            self.h_dec = nn.Parameter(self.h_dec, requires_grad=True)
            self.c_dec = nn.Parameter(self.c_dec, requires_grad=True)

    def forward(self, x: Tensor, cond: Tensor):
        batch_size = x.size(0)

        # img_c = 3
        # z_dim = 64
        # h_dim = 128
        # Hidden states initialization
        # h_enc: encoder hidden state size = (B, 128, 64, 64)
        # h_dec: decoder hidden state size = (B, 128, 64, 64)
        h_enc = self.h_enc.repeat(batch_size, 1, 1, 1)
        c_enc = self.c_enc.repeat(batch_size, 1, 1, 1)
        h_dec = self.h_enc.repeat(batch_size, 1, 1, 1)
        c_dec = self.c_dec.repeat(batch_size, 1, 1, 1)

        r = x.new_zeros((batch_size, self.c, self.h, self.w))
        kl = 0
        for _ in range(self.T):

            # Reconstruction error
            epsilon = x - r  # (B, 3, 64, 64)

            # Infer posterior density from hidden state (W//8)
            # (B, 3+3+128, 64, 64) ==> (B, 128, 64, 64)
            h_enc, c_enc = self.encoder(torch.cat([x, epsilon, h_dec], dim=1),
                                        h_enc, c_enc)

            # Prior
            p_mu, p_log_var = torch.chunk(self.prior(h_dec), 2, dim=1)
            p_std = torch.exp(p_log_var * 0.5)
            p_prior = Normal(p_mu, p_std)

            # Posterior distribution
            # (B, 128, 64, 64) ==> 2*(B, 64, 64, 64)
            q_mu, q_log_var = torch.chunk(self.posterior(h_enc), 2, dim=1)
            q_std = torch.exp(q_log_var * 0.5)
            q_posterior = Normal(q_mu, q_std)

            # Sample from posterior
            # (B, 64, 64, 64)
            z = q_posterior.rsample()

            # Send representation through decoder
            # cond: (B, CE), CE = x or y
            #       (B, CE, Zshape)
            z_shape = z.size()[:-2]
            cond_ = cond.clone().view(batch_size, -1, 1, 1)
            cond_ = cond_.contiguous().repeat(1, 1, *z_shape)
            h_dec, c_dec = self.decoder(torch.cat([z, r, cond_], dim=1), h_dec,
                                        c_dec)

            # write output
            # (B, 128, 64, 64) ==> 2*(B, 3, 64, 64)
            r_mu, r_log_var = torch.chunk(self.write(h_dec), 2, dim=1)
            r += r_mu

            # KL divergence
            kl += kl_divergence(q_posterior, p_prior)

        x_const = torch.sigmoid(r)
        r_std = torch.exp(r_log_var * 0.5)

        # calculate loss
        nll = -1 * torch.sum(Normal(x_const, r_std).log_prob(x))
        kl = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

        return x_const, kl, nll, r_std.detach().mean()

    def generate(self, x, cond):
        """
        Sample from the prior to generate a new
        datapoint.
        :param x: tensor representing shape of sample
        """
        batch_size = x.size(0)

        h_dec = self.h_dec.repeat(batch_size, 1, 1, 1)
        c_dec = self.c_dec.repeat(batch_size, 1, 1, 1)
        r = x.new_zeros((batch_size, self.c, self.h, self.w))

        for _ in range(self.T):
            # z = torch.randn(batch_size,
            #                 self.z_dim,
            #                 self.h,
            #                 self.w,
            #                 device=x.device)
            p_mu, p_log_var = torch.chunk(self.prior(h_dec), 2, dim=1)
            p_std = torch.exp(p_log_var * 0.5)
            z = Normal(p_mu, p_std).sample()

            cond_ = cond.clone().view(batch_size, -1, 1, 1)
            cond_ = cond_.contiguous().repeat(1, 1, self.h, self.w)
            h_dec, c_dec = self.decoder(torch.cat([z, r, cond_], dim=1), h_dec,
                                        c_dec)

            r_mu, _ = torch.chunk(self.write(h_dec), 2, dim=1)
            r += r_mu

        return torch.sigmoid(r)
