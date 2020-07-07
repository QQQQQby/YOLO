# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F


class YOLOv1Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
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

        for i in range(7, 15, 2):
            setattr(self, 'conv%d' % i, nn.Conv2d(512, 256, 1, stride=1, padding=0))
            setattr(self, 'batch_norm%d' % i, nn.BatchNorm2d(256))
            setattr(self, 'conv%d' % (i + 1), nn.Conv2d(256, 512, 3, stride=1, padding=1))
            setattr(self, 'batch_norm%d' % (i + 1), nn.BatchNorm2d(512))
        self.conv15 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.batch_norm15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.batch_norm16 = nn.BatchNorm2d(1024)
        self.pool4 = nn.MaxPool2d(2, 2)

        for i in range(17, 21, 2):
            setattr(self, 'conv%d' % i, nn.Conv2d(1024, 512, 1, stride=1, padding=0))
            setattr(self, 'batch_norm%d' % i, nn.BatchNorm2d(512))
            setattr(self, 'conv%d' % (i + 1), nn.Conv2d(512, 1024, 3, stride=1, padding=1))
            setattr(self, 'batch_norm%d' % (i + 1), nn.BatchNorm2d(1024))
        self.conv21 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.batch_norm21 = nn.BatchNorm2d(1024)
        self.conv22 = nn.Conv2d(1024, 1024, 3, stride=2, padding=1)
        self.batch_norm22 = nn.BatchNorm2d(1024)
        self.conv23 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.batch_norm23 = nn.BatchNorm2d(1024)
        self.conv24 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.batch_norm24 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024 * 7 * 7, 4096, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 7 * 7 * 30, bias=True)

    def forward(self, x):
        """448 * 448 * 3"""
        x = x.permute(0, 3, 1, 2).float()
        """3 * 448 * 448"""
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        x = self.pool1(x)
        """64 * 112 * 112"""
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        x = self.pool2(x)
        """192 * 56 * 56"""
        for i in range(3, 7):
            x = getattr(self, 'conv%d' % i)(x)
            x = getattr(self, 'batch_norm%d' % i)(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
        x = self.pool3(x)
        """512 * 28 * 28"""
        for i in range(7, 17):
            x = getattr(self, 'conv%d' % i)(x)
            x = getattr(self, 'batch_norm%d' % i)(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
        x = self.pool4(x)
        """1024 * 14 * 14"""
        for i in range(17, 25):
            x = getattr(self, 'conv%d' % i)(x)
            x = getattr(self, 'batch_norm%d' % i)(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
        """1024 * 7 * 7"""
        x = x.view(-1, 1024 * 7 * 7)
        """50176"""
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        """4096"""
        x = self.fc2(x)
        """1470"""
        x = x.view(-1, 7, 7, 30)
        """30 * 7 * 7"""
        return x


class TinyYOLOv1Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.batch_norm7 = nn.BatchNorm2d(1024)

        self.conv8 = nn.Conv2d(1024, 256, 3, stride=1, padding=1)
        self.batch_norm8 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 7 * 7, 7 * 7 * 30, bias=True)

    def forward(self, x):
        """448 * 448 * 3"""
        x = x.permute(0, 3, 1, 2).float()
        """3 * 448 * 448"""
        for i in range(1, 7):
            x = getattr(self, 'conv%d' % i)(x)
            x = getattr(self, 'batch_norm%d' % i)(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
            x = getattr(self, 'pool%d' % i)(x)
        """512 * 28 * 28"""
        for i in range(7, 9):
            x = getattr(self, 'conv%d' % i)(x)
            x = getattr(self, 'batch_norm%d' % i)(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
        """256 * 7 * 7"""
        x = x.view(-1, 256 * 7 * 7)
        """25088"""
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        """1470"""
        x = x.view(-1, 7, 7, 30)
        """30 * 7 * 7"""
        return x
