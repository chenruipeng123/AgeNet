'''
2023.2.26
跳跃链接处增加aspp
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from base.funtion import ASPP
from itertools import chain
# from funtion import ASPP
# from funtion import ASPP
# import utils.ASPP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return down_conv

# def x2conv(in_channels, out_channels, inner_channels=None):
#     inner_channels = out_channels // 2 if inner_channels is None else inner_channels
#     down_conv = nn.Sequential(
#         nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(inner_channels),
#         nn.ReLU(inplace=True),
#         # nn.GELU(),
#         nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         # nn.GELU()
#         nn.ReLU(inplace=True)
#     )
#     return down_conv

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)
        # self.aspp1 = ASPP(512, [6, 12, 18], 512)
        # self.aspp2 = ASPP(256, [6, 12, 18], 256)
        # self.aspp3 = ASPP(128, [6, 12, 18], 128)
        # self.aspp4 = ASPP(64, [6, 12, 18], 64)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)
        # x_copy = self.aspp1(x_copy)

        if (x.size(2) != x_copy.size(2) or x.size(3) != x_copy.size(3)):
            if interpolate:
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)), mode="bilinear", align_corners=True)
            else:
                diffy = x_copy.size()[2] - x.size()[2]
                diffx = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffx//2, diffx - diffx//2),
                              diffy//2, diffy - diffy//2)

        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=4, freeze_bn=False):
        super(UNet, self).__init__()

        self.start_conv = x2conv(in_channels, 64)
        self.down1 = Encoder(64, 128)
        self.down2 = Encoder(128, 256)
        self.down3 = Encoder(256, 512)
        self.down4 = Encoder(512, 1024)

        self.middle_conv = x2conv(1024, 1024)


        self.up1 = Decoder(1024, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)
        self.up4 = Decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self._initialize_weights()

        self.dropout = nn.Dropout(p=0.5)

        #ASPP

        if freeze_bn:
            self.freeze_bn()
    def _initialize_weights(self):
        num = 0
        for m in self.modules():
            num +=1
            # print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        print(f"depth: {num}")
    def forward(self, x):
        x1 = self.start_conv(x) # 64 256 256
        x2 = self.down1(x1) # 128 128 128
        x3 = self.down2(x2) # 256 64 64
        x4 = self.down3(x3) # 512 32 32
        x = self.middle_conv(self.down4(x4)) # 1024 16 16
        # x = self.dropout(x)

        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.final_conv(x)
        # x_sea = x[:, 6:7, :, :]
        # x_sea = torch.cat([x_sea,torch.zeros(x_sea.shape).to(device)], dim=1)
        #print(f"x: {x.shape}")
#         x = self.softmax(x)

        return x
# #
# input = torch.rand(1,1,256,256)
#
# model = UNet(num_classes=7)
#
# # output = model(input)
# #
# # print(output.shape)
# '''
# # @time:2023/2/26 9:44
# # Author:Tuan
# # @File:TT_dataset_3l.py
# '''
# import torch
# import numpy as np
# from torchvision import transforms
# from torchvision.transforms import ToTensor
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.functional import one_hot
# import imageio
# import glob
# import os
#
#
# class MyDataset(Dataset):
#     def __init__(self, images_path, labels_path, lab_sl, lab_line, Transform=None):
#         """"""
#         # 在这里写，获得所有image路径，所有label路径的代码，并将路径放在分别放在images_path_list和labels_path_list中
#         """"""
#         self.images_path_list = glob.glob(os.path.join(images_path, '*.tif'))
#         self.labels_path_list = glob.glob(os.path.join(labels_path, '*.tif'))
#         self.lab_line_path_list = glob.glob(os.path.join(lab_line, '*.tif'))
#         self.lab_sl_list = glob.glob(os.path.join(lab_sl, '*.tif'))
#         self.transform = ToTensor()
#
#     def __getitem__(self, index):
#         # self.images_path_list.sort()
#         # self.labels_path_list.sort()
#
#         image_path = self.images_path_list[index]
#         label_path = self.labels_path_list[index]
#         lab_line_path = self.lab_line_path_list[index]
#         lab_sl_path = self.lab_sl_list[index]
#
#         image = imageio.imread(image_path)
#         label = imageio.imread(label_path) - 1
#         label_line = imageio.imread(lab_line_path)
#         label_sl = imageio.imread(lab_sl_path)
#
#         image = torch.from_numpy(image)
#         image = torch.permute(image, [2, 0, 1])
#
#         label = torch.from_numpy(label)
#         label_line = torch.from_numpy(label_line)
#         label_sl = torch.from_numpy(label_sl)
#         label = torch.squeeze(label, 0)
#         label_line = torch.squeeze(label_line, 0)
#         label_sl = torch.squeeze(label_sl, 0)
#
#         # label = torch.squeeze(label, 0)
#         # print(label.shape)
#         # label = one_hot(label.long(), num_classes=10)
#         # label = torch.squeeze(label, 0)
#         # label = np.transpose(label, ( 2, 0, 1))
#
#         return image, label, label_sl, label_line
#
#     def __len__(self):
#         return len(self.images_path_list)
#
# class MyDataset_up(Dataset):
#     def __init__(self, labels_path, lab_line, Transform=None):
#         """"""
#         # 在这里写，获得所有image路径，所有label路径的代码，并将路径放在分别放在images_path_list和labels_path_list中
#         """"""
#         # self.images_path_list = glob.glob(os.path.join(images_path, '*.tif'))
#         self.labels_path_list = glob.glob(os.path.join(labels_path, '*.tif'))
#         self.lab_line_path_list = glob.glob(os.path.join(lab_line, '*.tif'))
#         # self.lab_sl_list = glob.glob(os.path.join(lab_sl, '*.tif'))
#         self.transform = ToTensor()
#
#     def __getitem__(self, index):
#         # self.images_path_list.sort()
#         # self.labels_path_list.sort()
#
#         # image_path = self.images_path_list[index]
#         label_path = self.labels_path_list[index]
#         lab_line_path = self.lab_line_path_list[index]
#         # lab_sl_path = self.lab_sl_list[index]
#
#         # image = imageio.imread(image_path)
#         label = imageio.imread(label_path) - 1
#         label_line = imageio.imread(lab_line_path)
#         # label_sl = imageio.imread(lab_sl_path)
#
#         # image = torch.from_numpy(image)
#         # image = torch.permute(image, [2, 0, 1])
#
#         label = torch.from_numpy(label)
#         label_line = torch.from_numpy(label_line)
#         # label_sl = torch.from_numpy(label_sl)
#         label = torch.unsqueeze(label, 0)
#         label_line = torch.squeeze(label_line, 0)
#         # label_sl = torch.squeeze(label_sl, 0)
#
#         # label = torch.squeeze(label, 0)
#         # print(label.shape)
#         # label = one_hot(label.long(), num_classes=10)
#         # label = torch.squeeze(label, 0)
#         # label = np.transpose(label, ( 2, 0, 1))
#
#         return  label.to(torch.float32),  label_line
#
#     def __len__(self):
#         return len(self.labels_path_list)
# # def main():
# imagePath = r"E:\yqj\try\code\torch\Train\Data\coastline\img"
# labelPath = r"E:\yqj\try\code\torch\Train\Data\coastline\lab_type"
# lab_sl = r"E:\yqj\try\code\torch\Train\Data\coastline\lab_SL"
# lab_line = r"E:\yqj\try\code\torch\Train\Data\coastline\lab_line"
# mydataset = MyDataset_up( labelPath,  lab_line)
#
# # dataset = MyDataset('.\data_npy\img_covid_poisson_glay_clean_BATCH_64_PATS_100.npy')
#
#
# Data = DataLoader(mydataset, batch_size=2, shuffle=False, pin_memory=True)
# for i, data in enumerate(Data):
#     img , lab = data
#     output = model.forward(img)
#     print(img.shape)
#     print(output.shape)
#
#
#
# if __name__ == '__main__':
#     main()





