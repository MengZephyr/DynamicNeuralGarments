import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
import math
from typing import List
import torch.nn.functional as F


def siren_uniform(tensor: torch.Tensor, mode: str='fan_in', c:float=6):
    fan = _calculate_correct_fan(tensor, mode)
    std = 1/math.sqrt(fan)
    bound = math.sqrt(c) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class Sine(nn.Module):
    def __init__(self, w0:float = 1.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x:torch.Tensor):
        return torch.sin(self.w0*x)


class SIREN_AutoEncoder(nn.Module):
    def __init__(self, inDim:int, outDim:int, layers:List[int], w0:float=1.0,
                 w0_ini:float=30.0, bias:bool=True, c:float=6):
        super(SIREN_AutoEncoder, self).__init__()
        self.En_layers = [nn.Conv2d(inDim, layers[0], kernel_size=3, stride=1, padding=1, bias=bias), Sine(w0=w0_ini)]
        for index in range(len(layers)-1):
            self.En_layers.extend([nn.Conv2d(layers[index], layers[index+1], kernel_size=3,
                                             stride=2, padding=1, bias=bias), Sine(w0=w0)])
        self.encoder = nn.Sequential(*self.En_layers)
        for m in self.encoder.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(self.encoder.state_dict()[m], mode='fan_in', c=c)

        self.midLayers = [nn.Conv2d(layers[-1], layers[-1], kernel_size=3, stride=1, padding=1, bias=bias),
                          Sine(w0=w0),
                          nn.Conv2d(layers[-1], layers[-1], kernel_size=3, stride=1, padding=1, bias=bias),
                          Sine(w0=w0)]
        self.latenNet = nn.Sequential(*self.midLayers)
        for m in self.latenNet.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(self.latenNet.state_dict()[m], mode='fan_in', c=c)

        self.De_layers = []
        numL = len(layers)
        for index in range(len(layers)-1):
            self.De_layers.extend([nn.ConvTranspose2d(layers[numL-1-index], layers[numL-2-index],
                                                      kernel_size=4, stride=2, padding=1, bias=bias), Sine(w0=w0)])
        self.De_layers.append(nn.Conv2d(layers[0], outDim, kernel_size=3, stride=1, padding=1, bias=bias))
        self.decoder = nn.Sequential(*self.De_layers)
        for m in self.decoder.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(self.decoder.state_dict()[m], mode='fan_in', c=c)

    def interpolate(self, x, a):
        assert x.size()[0] == 2
        ox = (1.-a) * x[0, :, :, :] + a * x[1, :, :, :]
        x = ox.unsqueeze(0)
        x = self.encoder(x)
        y = self.latenNet(x)
        x = x + y
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        y = self.latenNet(x)
        x = x + y
        x = self.decoder(x)
        return x


class SIREN(nn.Module):
    def __init__(self, inDim:int, outDim:int, layers:List[int], w0:float=1.0,
                 w0_ini:float=30.0, bias:bool=True, c:float=6):
        super(SIREN, self).__init__()
        self.layers = [nn.Conv2d(inDim, layers[0], kernel_size=3, stride=1, padding=1, bias=bias), Sine(w0=w0_ini)]
        for index in range(len(layers)-1):
            self.layers.extend([nn.Conv2d(layers[index], layers[index+1], kernel_size=3,
                                          stride=1, padding=1, bias=bias), Sine(w0=w0)])

        self.layers.append(nn.Conv2d(layers[-1], outDim, kernel_size=3, stride=1, padding=1, bias=bias))
        self.network = nn.Sequential(*self.layers)
        print(self.network)

        for m in self.network.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(self.network.state_dict()[m], mode='fan_in', c=c)

    def forward(self, X):
        return self.network(X)

