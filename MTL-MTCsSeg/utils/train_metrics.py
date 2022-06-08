import numpy as np
import os
import torch.nn as nn
import torch
from PIL import ImageOps, Image
from sklearn.metrics import confusion_matrix
from skimage import filters

from utils.evaluation_metrics3D import metrics_3d, Dice


def threshold(image):
    # t = filters.threshold_otsu(image, nbins=256)
    image[image >= 100] = 255
    image[image < 100] = 0
    return image


def numeric_score1(pred, gt):
    FP1 = np.float(np.sum((pred == 1) & (gt != 1)))
    FN1 = np.float(np.sum((pred != 1) & (gt == 1)))
    TP1 = np.float(np.sum((pred == 1) & (gt == 1)))
    TN1 = np.float(np.sum((pred != 1) & (gt != 1)))


    return FP1, FN1, TP1, TN1

def numeric_score2(pred, gt):
    FP2 = np.float(np.sum((pred == 2) & (gt != 2)))
    FN2 = np.float(np.sum((pred != 2) & (gt == 2)))
    TP2 = np.float(np.sum((pred == 2) & (gt == 2)))
    TN2 = np.float(np.sum((pred != 2) & (gt != 2)))


    return FP2, FN2, TP2, TN2

def numeric_score3(pred, gt):
    FP3 = np.float(np.sum((pred == 3) & (gt != 3)))
    FN3 = np.float(np.sum((pred != 3) & (gt == 3)))
    TP3 = np.float(np.sum((pred == 3) & (gt == 3)))
    TN3 = np.float(np.sum((pred != 3) & (gt != 3)))


    return FP3, FN3, TP3, TN3

def numeric_score4(pred, gt):
    FP4 = np.float(np.sum((pred == 4) & (gt != 4)))
    FN4 = np.float(np.sum((pred != 4) & (gt == 4)))
    TP4 = np.float(np.sum((pred == 4) & (gt == 4)))
    TN4 = np.float(np.sum((pred != 4) & (gt != 4)))


    return FP4, FN4, TP4, TN4


def metrics(pred, label, batch_size):
    pred = torch.argmax(pred, dim=1) # for CE Loss series
    outputs = (pred.data.cpu().numpy()).astype(np.uint8)
    labels = (label.data.cpu().numpy()).astype(np.uint8)
    #outputs = outputs.squeeze(1)  # for MSELoss()
    #labels = labels.squeeze(1)  # for MSELoss()
    #outputs = threshold(outputs)  # for MSELoss()

    Acc1, SEn1 , IoU1= 0., 0., 0.
    Acc2, SEn2, IoU2 = 0., 0., 0.
    Acc3, SEn3, IoU3 = 0., 0., 0.
    Acc4, SEn4, IoU4 = 0., 0., 0.
    for i in range(batch_size):
        img = outputs[i, :, :]
        gt = labels[i, :, :]
        acc1, sen1, iou1, acc2, sen2, iou2, acc3, sen3, iou3, acc4, sen4, iou4 = get_acc(img, gt)

        Acc1 += acc1
        SEn1 += sen1
        IoU1 += iou1

        Acc2 += acc2
        SEn2 += sen2
        IoU2 += iou2

        Acc3 += acc3
        SEn3 += sen3
        IoU3 += iou3

        Acc4 += acc4
        SEn4 += sen4
        IoU4 += iou4

    return Acc1, SEn1, IoU1, Acc2, SEn2, IoU2, Acc3, SEn3, IoU3, Acc4, SEn4, IoU4

def metrics3dmse(pred, label, batch_size):
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    outputs = outputs.squeeze(1)  # for MSELoss()
    labels = labels.squeeze(1)  # for MSELoss()
    outputs = threshold(outputs)  # for MSELoss()

    tp, fn, fp, IoU = 0, 0, 0, 0
    for i in range(batch_size):
        img = outputs[i, :, :, :]
        gt = labels[i, :, :, :]
        tpr, fnr, fpr, iou = metrics_3d(img, gt)
        # dcr = Dice(img, gt)
        tp += tpr
        fn += fnr
        fp += fpr
        IoU += iou
    return tp, fn, fp, IoU


def metrics3d(pred, label, batch_size):
    pred = torch.argmax(pred, dim=1)  # for CE loss series
    outputs = (pred.data.cpu().numpy() * 255).astype(np.uint8)
    labels = (label.data.cpu().numpy() * 255).astype(np.uint8)
    # outputs = outputs.squeeze(1)  # for MSELoss()
    # labels = labels.squeeze(1)  # for MSELoss()
    # outputs = threshold(outputs)  # for MSELoss()

    tp, fn, fp, IoU = 0, 0, 0, 0
    for i in range(batch_size):
        img = outputs[i, :, :, :]
        gt = labels[i, :, :, :]
        tpr, fnr, fpr, iou = metrics_3d(img, gt)
        # dcr = Dice(img, gt)
        tp += tpr
        fn += fnr
        fp += fpr
        IoU += iou
    return tp, fn, fp, IoU


def get_acc(image, label):
    #image = threshold(image)

    FP1, FN1, TP1, TN1 = numeric_score1(image, label)
    acc1 = (TP1 + TN1) / (TP1 + FN1 + TN1 + FP1 + 1e-10)
    sen1 = (TP1) / (TP1 + FN1 + 1e-10)
    iou1 = TP1 / (TP1 + FN1 + FP1 + 1e-10)

    FP2, FN2, TP2, TN2 = numeric_score2(image, label)
    acc2 = (TP2 + TN2) / (TP2 + FN2 + TN2 + FP2 + 1e-10)
    sen2 = (TP2) / (TP2 + FN2 + 1e-10)
    iou2 = TP2 / (TP2 + FN2 + FP2 + 1e-10)

    FP3, FN3, TP3, TN3 = numeric_score3(image, label)
    acc3 = (TP3 + TN3) / (TP3 + FN3 + TN3 + FP3 + 1e-10)
    sen3 = (TP3) / (TP3 + FN3 + 1e-10)
    iou3 = TP3 / (TP3 + FN3 + FP3 + 1e-10)

    FP4, FN4, TP4, TN4 = numeric_score4(image, label)
    acc4 = (TP4 + TN4) / (TP4 + FN4 + TN4 + FP4 + 1e-10)
    sen4 = (TP4) / (TP4 + FN4 + 1e-10)
    iou4 = TP4 / (TP4 + FN4 + FP4 + 1e-10)



    return acc1, sen1, iou1, acc2, sen2, iou2, acc3, sen3, iou3, acc4, sen4, iou4
