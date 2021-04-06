##
import torch
import torch.nn as nn

from layer import *

## network

class UNet(nn.Module):
    def __init__(self, nch, nker, norm="bnorm", learning_type="plain"):
        super(UNet, self).__init__()

        def UnPool(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True):
            return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)

        self.learning_type = learning_type

        # Contracting path
        self.enc1_1 = CBR2d(nch, 1 * nker, norm=norm)
        self.enc1_2 = CBR2d(1 * nker, 1 * nker, norm=norm)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(1 * nker, 2 * nker, norm=norm)
        self.enc2_2 = CBR2d(2 * nker, 2 * nker, norm=norm)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(2 * nker, 4 * nker, norm=norm)
        self.enc3_2 = CBR2d(4 * nker, 4 * nker, norm=norm)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(4 * nker, 8 * nker, norm=norm)
        self.enc4_2 = CBR2d(8 * nker, 8 * nker, norm=norm)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(8 * nker, 16 * nker, norm=norm)

        # Expansive path
        self.dec5_1 = CBR2d(16 * nker, 8 * nker, norm=norm)

        self.unpool4 = UnPool(8 * nker, 8 * nker)

        self.dec4_2 = CBR2d(2 * 8 * nker, 8 * nker, norm=norm)
        self.dec4_1 = CBR2d(8 * nker, 4 * nker, norm=norm)

        self.unpool3 = UnPool(4 * nker, 4 * nker)

        self.dec3_2 = CBR2d(2 * 4 * nker, 4 * nker, norm=norm)
        self.dec3_1 = CBR2d(4 * nker, 2 * nker, norm=norm)

        self.unpool2 = UnPool(2 * nker, 2 * nker)

        self.dec2_2 = CBR2d(2 * 2 * nker, 2 * nker, norm=norm)
        self.dec2_1 = CBR2d(2 * nker, 1 * nker, norm=norm)

        self.unpool1 = UnPool(1 * nker, 1 * nker)

        self.dec1_2 = CBR2d(2 * 1 * nker, 1 * nker, norm=norm)
        self.dec1_1 = CBR2d(1 * nker, 1 * nker, norm=norm)

        self.fc = nn.Conv2d(in_channels=1 * nker, out_channels=nch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        if self.learning_type == "plain":
            x = self.fc(dec1_1)
        elif self.learning_type == "residual":
            x = x + self.fc(dec1_1)

        return x
