"""
Reference:
https://arxiv.org/pdf/1506.04214.pdf

https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py
"""

# import math

import torch
import torch.nn as nn
# from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class HadamardproductLayer(nn.Module):
    def __init__(self, n: int, h: int, w: int):
        super(HadamardproductLayer, self).__init__()

        self.n = n
        self.h = h
        self.w = w

        self.weights = Parameter(torch.zeros(1, n, h, w))
        self.b = Parameter(torch.zeros(1, h))

    # def init_parameters(self):
    #     # initialize weigths and biase
    #     init.kaiming_uniform_(self.weights, a=math.sqrt(5))
    #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
    #     bound = 1 / math.sqrt(fan_in)
    #     init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        return x * self.weights + self.b

    def extra_repr(self) -> str:
        return f"n={self.n}, h={self.h}, w={self.w}"


class ConvLSTMCell(nn.Module):
    """
    2d convolutional long short-term memory (LSTM) cell.

    Replace FF layers with conv2d layer

    Parameters:


    """
    def __init__(self, input_w, input_channels, hidden_channels, kernel_size,
                 stride, padding):
        super(ConvLSTMCell, self).__init__()

        # assert hidden_channels % 2 == 0

        self.model_param = nn.Parameter(torch.empty(0))

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        self.Wxf = nn.Conv2d(self.input_channels,
                             self.hidden_channels,
                             bias=False,
                             **kwargs)

        self.Wxi = nn.Conv2d(self.input_channels,
                             self.hidden_channels,
                             bias=False,
                             **kwargs)

        self.Wxc = nn.Conv2d(self.input_channels,
                             self.hidden_channels,
                             bias=False,
                             **kwargs)

        self.Wxo = nn.Conv2d(self.input_channels,
                             self.hidden_channels,
                             bias=False,
                             **kwargs)

        # hidden state
        self.Whf = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             bias=False,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.Whi = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             bias=False,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.Whc = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             bias=False,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.Who = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             bias=False,
                             kernel_size=5,
                             padding=2,
                             stride=1)

        self.Wci = HadamardproductLayer(hidden_channels, input_w, input_w)
        self.Wcf = HadamardproductLayer(hidden_channels, input_w, input_w)
        self.Wco = HadamardproductLayer(hidden_channels, input_w, input_w)

        self.b_c = nn.Parameter(torch.zeros(1, input_w))

        self.nrm_i = nn.GroupNorm(1, hidden_channels)
        self.nrm_f = nn.GroupNorm(1, hidden_channels)
        self.nrm_c = nn.GroupNorm(1, hidden_channels)
        self.nrm_o = nn.GroupNorm(1, hidden_channels)

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.nrm_i(self.Wxi(x) + self.Whi(h) + self.Wci(c)))
        cf = torch.sigmoid(self.nrm_f(self.Wxf(x) + self.Whf(h) + self.Wcf(c)))
        cc = cf * c + ci * torch.tanh(
            self.nrm_f(self.Wxc(x) + self.Whc(h) + self.b_c))
        co = torch.sigmoid(
            self.nrm_o(self.Wxo(x) + self.Who(h) + self.Wco(cc)))
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):

        device = self.model_param.device

        h_0 = torch.zeros(batch_size,
                          hidden,
                          shape[0],
                          shape[1],
                          device=device)

        c_0 = torch.zeros(batch_size,
                          hidden,
                          shape[0],
                          shape[1],
                          device=device)

        return (Variable(h_0), Variable(c_0))
