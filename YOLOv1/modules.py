# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bathc_norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 192, 3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(192, 128, 1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)

        # self.conv7 =

    def forward(self, x):
        """3 * 448 * 448"""
        x = self.conv1(x)
        x = self.bathc_norm1(x)
        x = F.leaky_relu(x)
        x = self.pool1(x)
        """64 * 112 * 112"""
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.leaky_relu(x)
        x = self.pool2(x)
        """192 * 56 * 56"""
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = F.leaky_relu(x)
        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = F.leaky_relu(x)
        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = F.leaky_relu(x)
        x = self.pool3(x)
        """512 * 28 * 28"""


        return x
