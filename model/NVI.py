import glob
import os
import cv2
import numpy as np
from skimage import io
import numpy as np
import cv2

def ConfusionMatrix(numClass, imgPredict, Label):
    # 如果Label是png图像，我们需要将其转换为numpy数组
    if isinstance(Label, str):
        Label = np.array(Image.open(Label))

    # 如果imgPredict是一个颜色图像，我们将其转换为灰度图像
    if imgPredict.ndim == 3:
        imgPredict = np.dot(imgPredict[...,:3], [0.2989, 0.5870, 0.1140])

    # 返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask].astype(int) + imgPredict[mask].astype(int)
    count = np.bincount(label, minlength=numClass**2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix



def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA


def Precision(confusionMatrix):
    #  返回所有类别的精确率precision
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return precision


def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    return recall


def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score


def IntersectionOverUnion(confusionMatrix):
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix,
                   axis=1) + np.sum(confusionMatrix,
                                    axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU


def MeanIntersectionOverUnion(confusionMatrix):
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix,
                   axis=1) + np.sum(confusionMatrix,
                                    axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
        np.sum(confusionMatrix, axis=1) +
        np.sum(confusionMatrix, axis=0) -
        np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


def kappa(confusionMatrix):
    # 返回kappa系数
    pe_rows = np.sum(confusionMatrix, axis=0)
    pe_cols = np.sum(confusionMatrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusionMatrix) / float(sum_total)
    return (po - pe) / (1 - pe)

#################################################################

#  计算混淆矩阵及各精度参数


def cal_score():
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    # FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)
    kappa_score = kappa(confusionMatrix)

    return precision, recall, f1ccore, IoU, OA, mIOU, kappa_score


if __name__ == '__main__':
    label = r"/chenruipeng/GreenTide/test/1.png"

    pre = r"/chenruipeng/GreenTide/test/1.tif"
    ndvi_pre = io.imread(pre)
#    ndvi = -(ndvi_pre[0] - ndvi_pre[1]) / (ndvi_pre[0] + ndvi_pre[1])

    ndvi_max = ndvi_pre.max()
    print('ndvi_max='+str(ndvi_max))
    print('ndvi_min='+str(ndvi_pre.min()))
    max= 0
    maxi=0
    for i in range(0,int(2*100)):
        #  类别数目(包括背景)
        print(i/100.0)
        pre = r"/chenruipeng/GreenTide/test/1.tif"
        ndvi_pre = io.imread(pre)
        yuzhi = i/100.0
        classNum = 2
        y_pre=ndvi_pre

        y_pre[y_pre>yuzhi]=100
        y_pre[y_pre <= yuzhi] = 0
        #y_pre[y_pre==10000]=1
        #print(y_pre)
        #  类别颜色字典

        y_pre[y_pre > 0] = 1
    
        y_ture = cv2.imread(label, 0)
        y_ture[y_ture == 255] = 1
    
        predict_all = np.array(y_pre).astype(np.uint8)
        label_all = np.array(y_ture).astype(np.uint8)
    
        score = cal_score()
    
        # 创建得分文件
        # make_xlsx_file(score, pre)
    
        cell_name = ['精确度', '召回率', 'F1分数', 'IOU', '整体精度', 'mIOU', 'Kappa系数']
        # for i in range(len(cell_name)):
        #     print(cell_name[i])
        #     if i >= 4:
        #         print(round(score[i], 4))
        #     else:
        #         print(round(score[i][0], 4), ', ', round(score[i][1], 4))



        if round(score[2][1], 4)>max:
            max=round(score[2][1], 4)
            maxi = i
        print(round(score[2][1], 4))
        print(maxi)
        io.imsave(r"/chenruipeng/GreenTide/save_res2/ndvi/" + str(i)+'_f1='+str(round(score[2][1], 4)) + '.tif', y_pre)