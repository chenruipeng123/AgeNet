# -*- coding: UTF-8 -*-
"""
@Author  ：ChenRuipeng of SDUST
@Date    ：2023/11/13 15:46
"""
import os
import numpy as np
import torch
import torch.nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from dataset.TT_Dataset  import MyDataset   # 读取数据所用函数
from mdoel.unet import UNet
from mdoel.DeeplabV3Plus import Deeplabv3plus_res50, Deeplabv3plus_res101, Deeplabv3plus_vitbase
from mdoel.FCN_ResNet import FCN_ResNet
from mdoel.vit_model import vit_fcn_model
from mdoel.HRNet import HighResolutionNet
from mdoel.Upernet import UPerNet
from mdoel.segformer.segformer import SegFormer
from mdoel.pspnet.pspnet import PSPNet
from model.ABCNet import ABCNet
from model.AgeNet import densehrnet
from model.AlgaeNet import UNet2
from model.WDSR import MODEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#超参数
num_classes = 2
batch_size = 8
start_ep = 0
end_ep = 200
epoches = end_ep - start_ep
save_epoch = 10  #每隔save_epoch个epoch 保存一个模型
model_name = "abcnet"

#保存路径
# save_path = r"/chenruipeng/RedTideDetection/save_model_redtide/"+model_name+"/"
#save_path = r"/chenruipeng/RedTideDetection/save_model_redtide/"+model_name+"/"

save_path = r"/chenruipeng/GreenTide/save_model/"+model_name+"/"

if os.path.exists(save_path) == False:
    os.makedirs(save_path)

# 读取数据
# ADE数据集
# imagePath = r"/chenruipeng/RedTideDetection/data_3/ADEChallengeData2016/images/training"  #图像路径
# labelPath = r"/chenruipeng/RedTideDetection/data_3/ADEChallengeData2016/annotations/training"  #真值路径

# # 养殖区数据集
# imagePath = r"/chenruipeng/RedTideDetection/data_newest/img"  #图像路径
# labelPath = r"/chenruipeng/RedTideDetection/data_newest/label"  #真值路径

# 赤潮数据集
imagePath = r"/chenruipeng/GreenTide/new_data/imgs"  #图像路径
labelPath = r"/chenruipeng/GreenTide/new_data/masks"  #真值路径

Dataset = MyDataset(imagePath, labelPath)
# 划分训练集和验证集
train_len = int(0.9 * len(Dataset))
val_len = len(Dataset) - train_len
trainDataset, valDataset = random_split(Dataset, [train_len, val_len])

trainDatasetloader = DataLoader(trainDataset, batch_size, shuffle=True)
valDatasetloader = DataLoader(valDataset, batch_size, shuffle=True)
trainLen = len(trainDatasetloader)
valLen = len(valDatasetloader)
print(f'the lenth of trainDatasetloader: {trainLen}\n')
print(f'the lenth of valDatasetloader: {valLen}')

# 初始化早停参数
patience = 20
best_val_loss = float('inf')
counter = 0

#构建模型、优化器、损失
if model_name == "unet":
    model = UNet(num_classes=num_classes, in_channels=3).to(device)
elif model_name == "deeplabv3p":
    model = Deeplabv3plus_res50(num_classes=num_classes, os=16, pretrained=False).to(device)
elif model_name == "fcn_resnet":
    model = FCN_ResNet(num_classes=num_classes, backbone='resnet50').to(device)
elif model_name == "deeplabv3p_resnet101":
    model = Deeplabv3plus_res101(num_classes=num_classes, os=16, pretrained=False).to(device)
elif model_name == "deeplabv3p_vitbase":
    model = Deeplabv3plus_vitbase(num_classes=num_classes, image_size=256, os=16, pretrained=True).to(device)
elif model_name == "vit_fcn":
    model = vit_fcn_model(num_classes=num_classes, pretrained=True).to(device)
elif model_name == "hrnet":
    model = HighResolutionNet(num_classes=2).to(device)
elif model_name == "upernet":
    model = UPerNet(num_classes=num_classes,image_size=64).to(device)
elif model_name == "upernet_vit":
    model = UPerNet(num_classes=num_classes, image_size=256).to(device)
elif model_name == "segformer":
    model = SegFormer(num_classes=2).to(device)
elif model_name == "pspnet":
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, num_classes=num_classes, zoom_factor=1, use_ppm=True).to(device)
elif model_name == "abcnet":
    model = ABCNet(3, 2).to(device)
elif model_name == "densehrnet":
    model = densehrnet(3,2).to(device)
elif model_name == "algaenet":
    model = UNet2(3, 2).to(device)
elif model_name == "srsenet":
    model = MODEL(3,1).to(device)
else:
    print("请选择一个模型！")
#加载权重文件
weights_file = ""
if weights_file != "":
    model.load_state_dict(torch.load(weights_file))
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
loss = CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001, betas=(0.9, 0.95))

#开始训练
for epoch in range(epoches):
    # 训练阶段
    model.train()
    epoch = start_ep
    print(
        f"\n----------------------------------------------epoch: {epoch}----------------------------------------------")
    loss_total = 0
    for i,data in enumerate(trainDatasetloader):
        
        image, label = data
        # image, label = data_ma
        image = image.to(device)
        label = label.to(device)
        #梯度清零
        optimizer.zero_grad()
        output = model(image)
        loss_step = loss(output, label.long())
        print("\r train: epoch: {}, step: {}/{},  loss: {}  ".format(epoch, i, trainLen,loss_step ,end=''))
        loss_step.backward()
        optimizer.step()
        loss_total += loss_step
    ep = epoch
    loss_epoch = loss_total/trainLen
    print("\r epoch: {}, epoch_loss: {}".format(epoch, round(float(loss_epoch), 8)), end='')

    # 验证阶段
    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for i, data in enumerate(valDatasetloader):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss_step = loss(output, label.long())
            val_loss_total += loss_step
    val_loss_epoch = val_loss_total / len(valDatasetloader)
    print("\n\r epoch: {}, val_loss: {}".format(epoch, round(float(val_loss_epoch), 8)), end='')

    # # 早停检查
    # if val_loss_epoch < best_val_loss:
    #     best_val_loss = val_loss_epoch
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print("Early stopping")
    #         break
    #保存模型
    weight_path =save_path+"epcho_"+str(epoch)+"_loss_"+str(round(float(loss_epoch), 8))+".pt"
    if (epoch + 1) % save_epoch == 0 :
        torch.save(model.state_dict(),weight_path)
    # 最后10个epoch： 每个epoch保存一个
    start_ep = start_ep+1
