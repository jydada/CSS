#!/usr/bin/python

'''
Martin Kersner, m.kersner@gmail.com
2015/11/30

Evaluation metrics for image segmentation inspired by
paper Fully Convolutional Networks for Semantic Segmentation.
'''
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
import scipy.io as sio
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np


def eachFile(filepath1, filepath2, filepath3):
    # pathDir1 =  os.listdir(filepath1)
    # for allDir1 in pathDir1:
    pathDir2 = os.listdir(filepath1)
    # print(pathDir2)
    for allDir2 in pathDir2:
        # allDir2=allDir1[:-7]
        # allDir2 = allDir1[:]
        # child1 = os.path.join('%s%s' % (filepath1, allDir1))
        # child2 = os.path.join('%s%s%s' % (filepath2, allDir2,'.tif'))
        # child2 = os.path.join('%s%s%s' % (filepath2, allDir2,'_ind.mat'))
        # child2 = os.path.join('%s%s' % (filepath3, allDir1))
        # print child1#.decode('gbk')
        # print child2  # .decode('gbk')
        # im = Image.open(child1)
        # im1 = np.array(im)
        # im2 = sio.loadmat(child2)
        # im2 = Image.open(child1)
        # im3 = np.array(im2)
        # im3 = im2["Ic"]
        ###############
        # allDir2 = allDir1[:-9]
        # eval_name = os.path.join('%s%s' % (filepath1,allDir1))
        # gt_name = os.path.join('%s%s%s' % (filepath2,allDir2,'_ConfidenceMap.png'))
        # evalpic = Image.open(eval_name)
        # savename = os.path.join('%s%s' % (filepath3,allDir2))
        ################
        allDir1 = allDir2[:-4]
        # print(allDir1)
        eval_name = os.path.join(filepath1, allDir2)
        gt_name = os.path.join(filepath2, allDir2[:-4] + '.tif')
        evalpic = Image.open(eval_name)
        savename = os.path.join(filepath3, allDir1 + '.mat')
        ################
        gt = Image.open(gt_name)
        a = np.array(gt)
        im3 = np.resize(a, (512, 512))
        im1 = np.array(evalpic)
        c1 = pixel_accuracy(im1, im3)
        # print c1
        c2 = mean_accuracy(im1, im3)
        # print c2
        c3 = mean_IU(im1, im3)
        # print c3
        c4 = frequency_weighted_IU(im1, im3)
        # print c4
        t = []

        t.append(c1)
        t.append(c2)
        t.append(c3)
        t.append(c4)
        print(t)
        tt = {}
        tt['acc'] = t
        sio.savemat(savename, tt)
        # return (child1,child2)


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# bb='/home/zc/project/gl/Nuclei_detection_segmentation/results/seg_test_mask_conf/'

segpath = 'D:/CSNet/assets/EPST20/pred/images/'
gtpath = 'D:/CSNet/dataset/DRIVE2/val/masks/'
savepath = 'D:/CSNet/assets/EPST20/pred/'
eachFile(segpath, gtpath, savepath)
# im = Image.open(n)
# im1=np.array(im)
# im2=sio.loadmat(m)
# im3=im2["ind"]
# c1=pixel_accuracy(im1,im3)
# print c1
# c2=mean_accuracy(im1,im3)
# print c2
# c3=mean_IU(im1,im3)
# print c3
# c4=frequency_weighted_IU(im1,im3)
# print c4
