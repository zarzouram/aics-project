"""
Reference:
https://arxiv.org/pdf/1506.04214.pdf

https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py
"""

# import math

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    2d convolutional long short-term memory (LSTM) cell.

    Replace FF layers with conv2d layer

    Parameters:


    """

    def __init__(self, input_channels, hidden_channels, kernel_size, stride,
                 padding):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        self.Wxf = nn.Conv2d(self.input_dim, self.hidden_dim, **kwargs)
        self.Wxi = nn.Conv2d(self.input_dim, self.hidden_dim, **kwargs)
        self.Wxc = nn.Conv2d(self.input_dim, self.hidden_dim, **kwargs)
        self.Wxo = nn.Conv2d(self.input_dim, self.hidden_dim, **kwargs)

        # hidden state
        self.Whf = nn.Conv2d(self.hidden_dim,
                             self.hidden_dim,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.Whi = nn.Conv2d(self.hidden_dim,
                             self.hidden_dim,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.Whc = nn.Conv2d(self.hidden_dim,
                             self.hidden_dim,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.Who = nn.Conv2d(self.hidden_dim,
                             self.hidden_dim,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.nrm_i = nn.BatchNorm2d(hidden_channels)
        self.nrm_f = nn.BatchNorm2d(hidden_channels)
        self.nrm_c = nn.BatchNorm2d(hidden_channels)
        self.nrm_o = nn.BatchNorm2d(hidden_channels)

    def forward(self, x, h, c):
        ig = torch.sigmoid(self.nrm_i(self.Wxi(x) + self.Whi(h)))
        fg = torch.sigmoid(self.nrm_f(self.Wxf(x) + self.Whf(h)))
        og = torch.sigmoid(self.nrm_c(self.Wxo(x) + self.Who(h)))
        cg = torch.tanh(self.nrm_o(self.Wxc(x) + self.Whc(h)))

        cell = fg * c + ig * cg
        hidden = og * torch.tanh(cell)
        return hidden, cell
