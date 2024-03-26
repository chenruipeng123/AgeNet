import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import imageio
import glob
import os
import cv2
# from torchvision import transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, Transform=None):
        """"""
        # 在这里写，获得所有image路径，所有label路径的代码，并将路径放在分别放在images_path_list和labels_path_list中
        """"""
        self.images_path_list = sorted(glob.glob(os.path.join(images_path, '*.jpg')))
        self.labels_path_list = sorted(glob.glob(os.path.join(labels_path, '*.png')))
        self.transform = ToTensor()

    def __getitem__(self, index):

        image_path = self.images_path_list[index]
        label_path = self.labels_path_list[index]
        image = imageio.imread(image_path)
        label = imageio.imread(label_path)

#         B1, B2, B3 = cv2.split(image)
#         # #计算NDVI 和 NDWI
#         # NDVI = (B4 - B3) / (B4 + B3)
#         # NDWI = (B2 - B4) / (B2 + B4)

#         B1_normalization = ((B1 - np.min(B1)) / (np.max(B1) - np.min(B1)) * 1).astype('float32')
#         B2_normalization = ((B2 - np.min(B2)) / (np.max(B2) - np.min(B2)) * 1).astype('float32')
#         B3_normalization = ((B3 - np.min(B3)) / (np.max(B3) - np.min(B3)) * 1).astype('float32')
        # B4_normalization = ((B4 - np.min(B4)) / (np.max(B4) - np.min(B4)) * 1).astype('float32')
        # NDVI_normalization = ((NDVI - np.min(NDVI)) / (np.max(NDVI) - np.min(NDVI))).astype('float32')
        # NDWI_normalization = ((NDWI - np.min(NDWI)) / (np.max(NDWI) - np.min(NDWI))).astype('float32')

        # #去除近红外波段和蓝光波段+NDVI
        # image = cv2.merge([B2_normalization, B3_normalization,NDVI_normalization])
        # 去除近红外波段+NDVI
        # image = cv2.merge([B1_normalization,  B2_normalization, B3_normalization, NDVI_normalization])
        # 原始输入
        # image = cv2.merge([B1_normalization, B2_normalization, B3_normalization, B4_normalization])
        # #去除蓝光波段+NDVI
        # image = cv2.merge([B2_normalization, B3_normalization, B4_normalization, NDVI_normalization])
        #没有蓝光
        # image = cv2.merge([B2_normalization, B3_normalization, B4_normalization])
        # 没有近红外
        # image = cv2.merge([B1_normalization, B2_normalization, B3_normalization])
        # 原始输入 + NDVI
        # image = cv2.merge([B1_normalization, B2_normalization, B3_normalization, B4_normalization, NDVI_normalization])
        # #去除近红外波段和蓝光波段+NDVI+NDWI
        # image = cv2.merge([B2_normalization, B3_normalization,NDVI_normalization, NDWI_normalization])
        # #去除近红外波段和蓝光波段+NDWI
        # image = cv2.merge([B2_normalization, B3_normalization, NDWI_normalization])
        # #去除蓝光波段+NDVI+NDWI
        # image = cv2.merge([B2_normalization, B3_normalization, B4_normalization, NDVI_normalization, NDWI_normalization])
        # 去除近红外波段 + NDVI + NDWI
        # image = cv2.merge([B1_normalization, B2_normalization, B3_normalization, NDVI_normalization, NDWI_normalization])
        #红光+ndvi+ndwi
        # image = cv2.merge([ B3_normalization,NDVI_normalization, NDWI_normalization])
        # ndvi
#         image = cv2.merge([B1_normalization, B2_normalization, B3_normalization])
        # image = np.expand_dims(image, axis=2)
        image = np.array(image)
        label = np.array(label) / 255
        image = image.astype(float)
        label = label.astype(float)

        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label)
        label = torch.squeeze(label, 0)

        return image.float(), label

    def __len__(self):
        return len(self.images_path_list)


def main():
    imagePath = r"/chenruipeng/GreenTide/data/img"  # 图像路径
    labelPath = r"/chenruipeng/GreenTide/data/label"  # 真值路径
    mydataset = MyDataset(imagePath, labelPath)

    Data = DataLoader(mydataset, batch_size=1, shuffle=False, pin_memory=True)
    for i, data in enumerate(Data):
        if data is not None:
            img, lab = data
            print(img.shape)
            print(lab.shape)


if __name__ == '__main__':
    main()
