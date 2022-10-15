import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import struct
import copy
from layers import GraphConvolution

# adjust aitivation func: relu, leaky_relu, tanh, sigmoid...
# consider batch_normalization, dropout
# adjust hid, hid/2, ...
# if the last activation is tanh in G, the input pixel value should be normalized to [-1, 1]?


class Generator(nn.Module):
    def __init__(self, D_in, D_out, num_layers, z_dim):
        super(Generator, self).__init__()
        super(Encoder, self).__init__())
        self.conv1 = nn.Conv2d(in_channels=D_in, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=D_out, kernel_size=5, stride=1, padding=0)
        self.linear1 = torch.nn.Linear(D_out, D_out)
        self.lstm = nn.LSTM(num_layers * D_out + D_out, D_out, num_layers)
        self.linear2 = torch.nn.Linear(num_layers * D_out, D_out)

       
    def forward(self, x, memory, c):
        # input noise x <- (batch_size, pixel_num, init_dimension)
        # input adjacency matrix <- (batch_size, region_width * region_length, region_width * region_length)
        # input condition c <- (batch_size, pixel_num, condition_num)
        x = torch.cat((x, c), dim=2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        memory = memory.repeat(x.size(0), 1)
        x = torch.cat((x, memory), dim = 1)

        x = x.view(x.size(0), 1, -1)
        out, (h, c) = self.lstm(x)

        h = h.view(-1)
        h = F.relu(self.linear2(h))
        return h


class Discriminator(nn.Module):
    def __init__(self, infeat, hid, outfeat):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=D_in, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=D_out, kernel_size=5, stride=1, padding=0)
        self.linear1 = torch.nn.Linear(D_out + z_dim, D_out)
        self.linear2 = torch.nn.Linear(D_out, 1)

    def forward(self, x, adj, c):
        # input region x <- (batch_size, pixel_num, feature_num)
        # input adjacency matrix <- (batch_size, region_width * region_length, region_width * region_length)
        # input condition c <- (batch_size, pixel_num, condition_num)
        x = torch.cat((x, c), dim=2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1)
        out = torch.cat((x, z))

        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return out
