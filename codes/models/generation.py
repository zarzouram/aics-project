"""
This code is edited from
Ref:
https://github.com/wohlert/generative-query-network-pytorch/tree/32f32632462d6796ecbc749a6617d3f3c0571b80

"""
import torch
from torch import Tensor

from torch import nn
from torch.distributions import Normal, kl_divergence

from codes.layers.conv_lstm_simple import ConvLSTMCell


class DRAW(nn.Module):

    def __init__(self,
                 imw: int,
                 imh: int,
                 imc: int,
                 iter_num: int,
                 h_dim: int,
                 z_dim: int,
                 cond_dim: int,
                 initc_tunning: bool = False):
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

        self.h = imh
        self.w = imw
        self.img_shape = (imw, imh)
        self.c = imc

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.T = iter_num

        self.initc_tunning = initc_tunning

        # Recurrent encoder/decoder models
        kwargs = dict(kernel_size=5, stride=1, padding=2)
        # kwargs = dict(kernel_size=3, stride=1, padding=1)

        self.encoder = ConvLSTMCell(imc * 2 + h_dim, h_dim, **kwargs)
        self.decoder = ConvLSTMCell(imc + z_dim + cond_dim, h_dim, **kwargs)

        # write outputs
        self.write = nn.Conv2d(h_dim, 2 * imc, kernel_size=1, stride=1)

        # parameters of distributions
        self.posterior = nn.Conv2d(h_dim, 2 * z_dim, **kwargs)
        self.prior = nn.Conv2d(h_dim, 2 * h_dim, **kwargs)

        self.init_hidden()
        self.init_weights()

    def init_hidden(self):

        self.h_enc = torch.zeros(1, self.h_dim, *self.img_shape)
        self.c_enc = torch.zeros(1, self.h_dim, *self.img_shape)
        self.h_dec = torch.zeros(1, self.h_dim, *self.img_shape)
        self.c_dec = torch.zeros(1, self.h_dim, *self.img_shape)

        if self.initc_tunning:
            self.h_enc = nn.Parameter(self.h_enc, requires_grad=True)
            self.c_enc = nn.Parameter(self.c_enc, requires_grad=True)
            self.h_dec = nn.Parameter(self.h_dec, requires_grad=True)
            self.c_dec = nn.Parameter(self.c_dec, requires_grad=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, cond: Tensor):
        batch_size = x.size(0)

        # img_c = 3
        # z_dim = 128
        # h_dim = 128

        # Hidden states initialization
        # h_enc: encoder hidden state size = (B, h_dim, 32, 32)
        # h_dec: decoder hidden state size = (B, img_c, 32, 32)
        h_enc = self.h_enc.repeat(batch_size, 1, 1, 1)
        c_enc = self.c_enc.repeat(batch_size, 1, 1, 1)
        h_dec = self.h_dec.repeat(batch_size, 1, 1, 1)
        c_dec = self.c_dec.repeat(batch_size, 1, 1, 1)

        r = x.new_zeros((batch_size, self.c, self.h, self.w))
        kl = 0
        for _ in range(self.T):

            # Reconstruction error
            epsilon = x - r  # (B, 3, 32, 32)

            # Infer posterior density from hidden state (W//8)
            # (B, 3+3+h_dim, 32, 32) ==> (B, h_dim, 32, 32)
            h_enc, c_enc = self.encoder(torch.cat([x, epsilon, h_dec], dim=1),
                                        h_enc, c_enc)

            # Prior
            # (B, 3, 32, 32) ==> (B, 2*3, 32, 32)
            p_mu, p_log_var = torch.chunk(self.prior(h_dec), 2, dim=1)
            p_std = torch.exp(p_log_var * 0.5)
            prior = Normal(p_mu, p_std)

            # Posterior distribution
            # (B, h_dim, 32, 32) ==> (B, 2*z_dim, 32, 32)
            q_mu, q_log_var = torch.chunk(self.posterior(h_enc), 2, dim=1)
            q_std = torch.exp(q_log_var * 0.5)
            posterior = Normal(q_mu, q_std)

            # Sample from posterior
            # (B, z_dim, 32, 32)
            z = posterior.rsample()

            # Send representation through decoder
            # cond:  (B, CE)
            # repeat (B, CE, z_dim)
            z_shape = z.size()[-2:]
            cond_ = cond.clone().view(batch_size, -1, 1, 1)
            cond_ = cond_.contiguous().repeat(1, 1, *z_shape)
            h_dec, c_dec = self.decoder(torch.cat([z, r, cond_], dim=1), h_dec,
                                        c_dec)

            # write output
            # (B, h_dim, 32, 32) ==> 2*(B, 3, 32, 32)
            r_mu, r_log_var = torch.chunk(self.write(h_dec), 2, dim=1)
            r += r_mu

            # KL divergence
            kl += kl_divergence(posterior, prior)

        x_const = torch.sigmoid(r)
        r_std = torch.exp(r_log_var * 0.5)

        # calculate loss
        nll = -1 * (torch.sum(Normal(x_const, r_std).log_prob(x)))
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
            cond_ = cond_.contiguous().repeat(1, 1, *z.size()[-2:])
            h_dec, c_dec = self.decoder(torch.cat([z, r, cond_], dim=1), h_dec,
                                        c_dec)

            r_mu, _ = torch.chunk(self.write(h_dec), 2, dim=1)
            r += r_mu

        return torch.sigmoid(r)
