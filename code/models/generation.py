"""
This code is edited from
Ref:
https://github.com/Natsu6767/Generating-Devanagari-Using-DRAW/blob/master/draw_model.py

"""

import torch
from torch import Tensor
from torch import nn
import torch.distributions as Dist
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
        read_N: int,
        write_N: int,
        encoder_size: int,
        decoder_size: int,
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
        read_N:         int
                        Image grid dimension (N x N)
        write_N:        int
                        Image grid dimension (N x N)
        encoder_size:   int
                        Encoder LSTM hidden size
        decoder_size:   int
                        Decoder LSTM hidden size
        z_size:         int
                        latent space size
        cond_size:      int
                        The conditioning variable size (r + vwp)
        """
        super(DRAW, self).__init__()

        self.T = iter_num
        self.A = img_w
        self.B = img_h
        self.z_size = z_size
        self.read_N = read_N
        self.write_N = write_N
        self.enc_size = encoder_size
        self.dec_size = decoder_size
        self.channel = img_c

        self.model_param = nn.Parameter(torch.empty(0))

        # Stores the generated image for each time step.
        self.cs = [0] * self.T
        self.x_reconst = [0] * self.T

        # To store appropriate values used for calculating the latent loss
        # (KL-Divergence loss)
        self.logsigmas = [0] * self.T
        self.sigmas = [0] * self.T
        self.mus = [0] * self.T

        self.encoder = nn.LSTMCell(
            2 * self.read_N * self.read_N * self.channel + self.dec_size,
            self.enc_size)

        # To get the mean and standard deviation for the distribution of z.
        self.fc_mu = nn.Linear(self.enc_size, self.z_size)
        self.fc_sigma = nn.Linear(self.enc_size, self.z_size)

        self.decoder = nn.LSTMCell(self.z_size + cond_size, self.dec_size)

        self.fc_write = nn.Linear(self.dec_size,
                                  self.write_N * self.write_N * self.channel)

        # To get the attention parameters. 5 in total.
        self.fc_attention = nn.Linear(self.dec_size, 5)

    def forward(self, x: Tensor, cond: Tensor):
        self.batch_size = x.size(0)
        self.device = self.model_param.device
        # requires_grad should be set True to allow backpropagation of the
        # gradients for training.
        h_enc_prev = torch.zeros(self.batch_size,
                                 self.enc_size,
                                 requires_grad=True,
                                 device=self.device)
        h_dec_prev = torch.zeros(self.batch_size,
                                 self.dec_size,
                                 requires_grad=True,
                                 device=self.device)

        enc_state = torch.zeros(self.batch_size,
                                self.enc_size,
                                requires_grad=True,
                                device=self.device)
        dec_state = torch.zeros(self.batch_size,
                                self.dec_size,
                                requires_grad=True,
                                device=self.device)
        c_prev = torch.zeros(self.batch_size,
                             self.B * self.A * self.channel,
                             requires_grad=True,
                             device=self.device)

        for t in range(self.T):
            # Equation 3.
            x_hat = x - torch.sigmoid(c_prev)

            # Equation 4.
            # Get the N x N glimpse.
            r_t = self.read(x, x_hat, h_dec_prev)
            # Equation 5.
            h_enc, enc_state = self.encoder(
                torch.cat((r_t, h_dec_prev), dim=1), (h_enc_prev, enc_state))
            # Equation 6.
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(
                h_enc)
            # Equation 7. | The conditioning variable is concatenated with the
            # sampled latent z according to A2 in arXiv:1807.01670v2
            h_dec, dec_state = self.decoder(torch.cat((z, cond), dim=1),
                                            (h_dec_prev, dec_state))
            # Equation 8.
            self.cs[t] = c_prev + self.write(h_dec)

            h_enc_prev = h_enc
            h_dec_prev = h_dec
            c_prev = self.cs[t - 1 if t > 0 else t]

        return z

    def read(self, x, x_hat, h_dec_prev):
        # Using attention
        (Fx, Fy), gamma = self.attn_window(h_dec_prev, self.read_N)

        def filter_img(img, Fx, Fy, gamma):
            Fxt = Fx.transpose(self.channel, 2)
            if self.channel == 3:
                img = img.view(-1, 3, self.B, self.A)
            elif self.channel == 1:
                img = img.view(-1, self.B, self.A)

            # Equation 27.
            glimpse = torch.matmul(Fy, torch.matmul(img, Fxt))
            glimpse = glimpse.view(-1,
                                   self.read_N * self.read_N * self.channel)

            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)

        return torch.cat((x, x_hat), dim=1)
        # No attention
        # return torch.cat((x, x_hat), dim=1)

    def write(self, h_dec):
        # Using attention
        # Equation 28.
        w = self.fc_write(h_dec)
        if self.channel == 3:
            w = w.view(self.batch_size, 3, self.write_N, self.write_N)
        elif self.channel == 1:
            w = w.view(self.batch_size, self.write_N, self.write_N)

        (Fx, Fy), gamma = self.attn_window(h_dec, self.write_N)
        Fyt = Fy.transpose(self.channel, 2)

        # Equation 29.
        wr = torch.matmul(Fyt, torch.matmul(w, Fx))
        wr = wr.view(self.batch_size, self.B * self.A * self.channel)

        return wr / gamma.view(-1, 1).expand_as(wr)
        # No attention
        # return self.fc_write(h_dec)

    def sampleQ(self, h_enc):
        epsilon = torch.randn(self.batch_size, self.z_size, device=self.device)

        # Equation 1.
        mu = self.fc_mu(h_enc)
        # Equation 2.
        log_sigma = self.fc_sigma(h_enc)
        sigma = torch.exp(log_sigma)

        z = mu + epsilon * sigma

        return z, mu, log_sigma, sigma

    def attn_window(self, h_dec, N):
        # Equation 21.
        params = self.fc_attention(h_dec)
        gx_, gy_, log_sigma_2, log_delta_, log_gamma = params.split(1, 1)

        # Equation 22.
        gx = (self.A + 1) / 2 * (gx_ + 1)
        # Equation 23
        gy = (self.B + 1) / 2 * (gy_ + 1)
        # Equation 24.
        delta = (max(self.A, self.B) - 1) / (N - 1) * torch.exp(log_delta_)
        sigma_2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma_2, delta, N), gamma

    def filterbank(self, gx, gy, sigma_2, delta, N, epsilon=1e-8):
        grid_i = torch.arange(
            start=0.0,
            end=N,
            device=self.device,
            requires_grad=True,
        ).view(1, -1)

        # Equation 19.
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta
        # Equation 20.
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta

        a = torch.arange(0.0, self.A, device=self.device,
                         requires_grad=True).view(1, 1, -1)
        b = torch.arange(0.0, self.B, device=self.device,
                         requires_grad=True).view(1, 1, -1)

        mu_x = mu_x.view(-1, N, 1)
        mu_y = mu_y.view(-1, N, 1)
        sigma_2 = sigma_2.view(-1, 1, 1)

        # Equations 25 and 26.
        Fx = torch.exp(-torch.pow(a - mu_x, 2) / (2 * sigma_2))
        Fy = torch.exp(-torch.pow(b - mu_y, 2) / (2 * sigma_2))

        Fx = Fx / (Fx.sum(2, True).expand_as(Fx) + epsilon)
        Fy = Fy / (Fy.sum(2, True).expand_as(Fy) + epsilon)

        if self.channel == 3:
            Fx = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
            Fx = Fx.repeat(1, 3, 1, 1)

            Fy = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
            Fy = Fy.repeat(1, 3, 1, 1)

        return Fx, Fy

    def loss(self, x: Tensor, cond: Tensor):
        self.forward(x, cond)
        # Kullback-Leibler divergence of latent prior distribution-Equation 10
        zero = torch.zeros_like(torch.stack(self.mus))
        one = torch.ones_like(torch.stack(self.sigmas))
        p_prior = Dist.Normal(zero, one)
        q_z = Dist.Normal(torch.stack(self.mus), torch.stack(self.sigmas))
        lz = Dist.kl.kl_divergence(q_z, p_prior).sum(dim=[-2, -1]).mean(0)

        # Reconstruction loss - Equation 10
        batch_size = x.size(0)
        img_size = [self.B, self.A]
        x_prime = self.cs[-1].view(batch_size, -1, *img_size)
        x_ = x.view(batch_size, -1, *img_size)

        x_recon = Dist.Bernoulli(logits=x_prime)
        lx = -x_recon.log_prob(x_).sum(dim=[1, 2, 3]).mean(0)

        draw_net_loss = lz + lx

        return draw_net_loss
