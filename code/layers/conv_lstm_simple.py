"""
Reference:
https://arxiv.org/pdf/1506.04214.pdf

https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py
"""

# import math

import torch
import torch.nn as nn

from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    2d convolutional long short-term memory (LSTM) cell.

    Replace FF layers with conv2d layer

    Parameters:


    """
    def __init__(self, function, input_w, input_channels, hidden_channels,
                 kernel_size, stride, padding):
        super(ConvLSTMCell, self).__init__()

        # assert hidden_channels % 2 == 0

        self.model_param = nn.Parameter(torch.empty(0))

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        if function == "encode":
            conv_fn = nn.Conv2d
        else:
            conv_fn = nn.ConvTranspose2d

        self.Wxf = conv_fn(self.input_channels,
                           self.hidden_channels,
                           **kwargs)

        self.Wxi = conv_fn(self.input_channels,
                           self.hidden_channels,
                           **kwargs)

        self.Wxc = conv_fn(self.input_channels,
                           self.hidden_channels,
                           **kwargs)

        self.Wxo = conv_fn(self.input_channels,
                           self.hidden_channels,
                           **kwargs)

        # hidden state
        self.Whf = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.Whi = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.Whc = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.Who = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.nrm_i = nn.GroupNorm(1, hidden_channels)
        self.nrm_f = nn.GroupNorm(1, hidden_channels)
        self.nrm_c = nn.GroupNorm(1, hidden_channels)
        self.nrm_o = nn.GroupNorm(1, hidden_channels)

    def forward(self, x, h, c):
        ig = torch.sigmoid(self.nrm_i(self.Wxi(x) + self.Whi(h)))
        fg = torch.sigmoid(self.nrm_f(self.Wxf(x) + self.Whf(h)))
        og = torch.sigmoid(self.nrm_c(self.Wxo(x) + self.Who(h)))
        cg = torch.tanh(self.nrm_o(self.Wxc(x) + self.Whc(h)))

        cell = fg * c + ig * cg
        hidden = og * torch.tanh(cell)
        return hidden, cell

    def init_hidden(self, batch_size, channel_size, shape):

        device = self.model_param.device

        h_0 = torch.zeros(batch_size,
                          channel_size,
                          shape[0],
                          shape[1],
                          device=device)

        c_0 = torch.zeros(batch_size,
                          channel_size,
                          shape[0],
                          shape[1],
                          device=device)

        return (Variable(h_0), Variable(c_0))
