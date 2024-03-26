""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn

from torchsummary import summary
# 原版


class l2_norm(nn.Module):
    def __init__(self,axit=1):
        super(l2_norm, self).__init__()
        self.axit = axit
    def forward(self,input):
        norm = torch.norm(input,2,self.axit,True)
        output = torch.div(input, norm)
        return output
# 这里假设input.shape = (2, 10) 最后得到的结果是每一行得到的每个值平方和为1


class conv33res(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(conv33res, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim,
                                            kernel_size=3, stride=1,padding=1, bias=False),
                                  nn.ReLU(inplace=True))
    def forward(self, x):
        x1 = self.conv(x)
        output = x1+x
        return output

class conv33re(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(conv33re, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim,
                                            kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv(x)
        output = x1
        return output


class conv11sig(nn.Module):
    def __init__(self, in_dim):
        super(conv11sig, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim,
                                            kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_dim):
        super(down, self).__init__()
        self.conv = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_dim):
        super(down, self).__init__()
        self.up = nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv(x)
        return x
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class bn(nn.Module):
    def __init__(self,dim=1):
        super(bn, self).__init__()
        self.l2 = l2_norm(dim)
    def forward(self, x):
        x =  self.l2(x)
        return x

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = nn.Sequential(
            DoubleConv(n_channels, 64),
            conv33re(64,64),
        )
        self.bn = l2_norm(1)
        self.down1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            conv33res(64,64),
            conv33res(64,64),
            bn(),
            conv33re(64,128)
        )
        self.down2 = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(
            conv33res(128,128),
            conv33res(128,128),
            bn(),
            conv33re(128,256)
        )
        self.down3 = nn.MaxPool2d(2)
        self.mid = nn.Sequential(
            conv33res(256,256),
            conv33res(256,256),
            bn(),
        )
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.block4 = nn.Sequential(
            conv33re(512,256),
            conv33re(256,256),
            bn(),
        )
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.block5 = nn.Sequential(
            conv33re(256+128,128),
            conv33re(128,128),
            bn(),
        )
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.block6 = nn.Sequential(
            conv33re(64 + 128, 64),
            conv33re(64, 64),
            bn(),
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.bn(x1)
        down1 = self.down1(x1)
        x2 = self.block2(down1)
        down2 = self.down2(x2)
        x3 = self.block3(down2)
        down3 = self.down3(x3)
        mid = self.mid(down3)
        mid_up = self.up1(mid)
        x4 = torch.cat([mid_up,x3],dim=1)
        x4 = self.block4(x4)
        up2 = self.up2(x4)
        x5 = torch.cat([x2,up2],dim=1)
        x5 = self.block5(x5)
        up3 = self.up3(x5)
        x6 = torch.cat([up3,x1],dim=1)
        x6 = self.block6(x6)
        out = self.outconv(x6)
        return out
if __name__ == '__main__':
#     img = torch.randn(1, 3, 256, 256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=UNet2(n_channels=3, n_classes=2, bilinear=False)
#     output = model.forward(img)
#     print(output.shape)
    summary(model, (3, 64, 64), device="cpu")