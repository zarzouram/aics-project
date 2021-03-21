"""
Reference:
https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    2d convolutional long short-term memory (LSTM) cell.

    Replace FF layers with conv2d layer

    Parameters:


    """
    def __init__(self, input_channels, hidden_channels, kernel_size, stride,
                 padding):
        super(ConvLSTMCell, self).__init__()

        # assert hidden_channels % 2 == 0

        self.model_param = nn.Parameter(torch.empty(0))

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels,
                             **kwargs)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels,
                             **kwargs)

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels,
                             **kwargs)

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels,
                             **kwargs)

        # hidden state
        h_padding = int((kernel_size - 1) / 2)
        self.Whf = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             kernel_size=kernel_size,
                             padding=h_padding,
                             stride=1)

        self.Whi = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             kernel_size=kernel_size,
                             padding=h_padding,
                             stride=1)

        self.Whc = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             kernel_size=kernel_size,
                             padding=h_padding,
                             stride=1)

        self.Who = nn.Conv2d(self.hidden_channels,
                             self.hidden_channels,
                             kernel_size=kernel_size,
                             padding=h_padding,
                             stride=1)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):

        device = self.model_param.device

        Wci = nn.Parameter(torch.zeros(1, hidden, shape[0] // 2,
                                       shape[1] // 2),
                           requires_grad=True)
        Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0] // 2,
                                       shape[1] // 2),
                           requires_grad=True)
        Wco = nn.Parameter(torch.zeros(1, hidden, shape[0] // 2,
                                       shape[1] // 2),
                           requires_grad=True)

        self.Wci = Wci.to(device)
        self.Wcf = Wcf.to(device)
        self.Wco = Wco.to(device)

        h_0 = torch.zeros(batch_size,
                          hidden,
                          shape[0] // 2,
                          shape[1] // 2,
                          device=device)

        c_0 = torch.zeros(batch_size,
                          hidden,
                          shape[0] // 2,
                          shape[1] // 2,
                          device=device)

        return (Variable(h_0), Variable(c_0))
