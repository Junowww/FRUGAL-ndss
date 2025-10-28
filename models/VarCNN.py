import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils import weight_norm

class DilatedBasic1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=(1, 1)):
        super(DilatedBasic1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=(kernel_size - 1) // 2,
                               dilation=dilation[0])
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) // 2,
                               dilation=dilation[1])
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, padding=0)

    def forward(self, x):
        residual = x.clone()
        out = self.conv1(x)
        out = F.relu(self.bn1(out), inplace=False)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), inplace=False)
        # shortcut = self.shortcut(residual)
        # out += shortcut
        out = F.relu(out, inplace=False)
        return out

class VarCNN(nn.Module):
    def __init__(self, num_classes):
        super(VarCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            DilatedBasic1D(64, 64, stride=1),
            DilatedBasic1D(64, 64)
        )

        self.layer2 = nn.Sequential(
            DilatedBasic1D(64, 128, stride=2),
            DilatedBasic1D(128, 128)
        )

        self.layer3 = nn.Sequential(
            DilatedBasic1D(128, 256, stride=2),
            DilatedBasic1D(256, 256)
        )

        self.layer4 = nn.Sequential(
            DilatedBasic1D(256, 512, stride=2),
            DilatedBasic1D(512, 512)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=False)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.softmax(x)
        return x

