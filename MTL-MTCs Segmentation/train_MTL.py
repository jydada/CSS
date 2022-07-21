"""
Training script for CS-Net
"""
import os
import random
import warnings
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import visdom
import numpy as np
from model.csnet import CSNet
from dataloader.drive import Data
from utils.train_metrics import metrics
from utils.visualize import init_visdom_line, update_lines
from utils.dice_loss_single_class import dice_coeff_loss

#1. set random.seed and cudnn performance
random.seed(888)
np.random.seed(888)
torch.manual_seed(888)
torch.cuda.manual_seed_all(888)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

args = {
    'root'      : '',
    'data_path' : 'dataset/DRIVE6/',
    'pretext_data_path' : 'dataset/Prostate lumen segmentation/',
    'epochs'    : 100,
    'lr'        : 0.0001,
    'snapshot'  : 10,
    'test_step' : 1,
    'ckpt_path' : 'checkpoint-Prostate lumen segmentation/80/',
    'batch_size': 6,
}

# # Visdom---------------------------------------------------------
X, Y = 0, 0.5  # for visdom
x_acc, y_acc = 0, 0
x_sen, y_sen = 0, 0
env, panel = init_visdom_line(X, Y, title='Train Loss', xlabel="iters", ylabel="loss")
env1, panel1 = init_visdom_line(x_acc, y_acc, title="Accuracy", xlabel="iters", ylabel="accuracy")
env2, panel2 = init_visdom_line(x_sen, y_sen, title="Sensitivity", xlabel="iters", ylabel="sensitivity")
# # ---------------------------------------------------------------

def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    torch.save(net, args['ckpt_path'] + 'CS_Net_DRIVE_' + str(iter) + '.pkl')
    print('--->saved model:{}<--- '.format(args['root'] + args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    # set the channels to 3 when the format is RGB, otherwise 1.
    net = CSNet(classes=5, channels=3).cuda()
    net = nn.DataParallel(net, device_ids=[0, 1]).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=0.0005)
    #critrion = nn.MSELoss().cuda()
    critrion = nn.CrossEntropyLoss().cuda()
    print("---------------start training------------------")
    # load train dataset # repeat train dataset due to pretext_data.__len__()//train_data.__len__()
    train_data = Data(args['data_path'], train=True)
    # batchs_data = DataLoader(train_data, batch_size=5, num_workers=2, shuffle=True)

    pretext_data = Data(args['pretext_data_path'], train=True)
    pretext_batchs_data = DataLoader(pretext_data, batch_size=6, num_workers=2, shuffle=True)
    print(pretext_data.__len__()//train_data.__len__())
    ii=pretext_data.__len__() // train_data.__len__()
    train_data = []
    for i in range(ii*6//6):
        print(i)
        train_data1 = Data(args['data_path'], train=True)
        train_data.extend(train_data1)
    batchs_data = DataLoader(train_data, batch_size=6, num_workers=2, shuffle=True)

    iters = 1
    accuracy = 0.
    sensitivty = 0.
    IOU = 0.
    for epoch in range(args['epochs']):
        net.train()
        for idx, (batch, batch2) in enumerate(zip(batchs_data, pretext_batchs_data)):
            image = batch[0].cuda()
            label = batch[1].cuda().long()
            image2 = batch2[0].cuda()
            label2 = batch2[1].cuda().long()
            optimizer.zero_grad()
            pred = net(image)
            pred2 = net(image2)
            # pred = pred.squeeze_(1)
            #pred = torch.argmax(pred, dim=1)
            #label = label.squeeze(1)
            loss = critrion(pred, label)
            loss2 = critrion(pred2, label2)
            #loss2 = dice_coeff_loss(pred, label)
            loss = loss + loss2
            loss.backward()
            optimizer.step()
            acc1, sen1, iou1, acc2, sen2, iou2, acc3, sen3, iou3, acc4, sen4, iou4 = metrics(pred, label, pred.shape[0])
            print('[{0:d}:{1:d}] --- loss:{2:.10f}\tacc1:{3:.4f}\tsen1:{4:.4f}\tiou1:{5:.4f}'.format(epoch + 1,
                                                                                                     iters, loss.item(),
                                                                                                     acc1 / pred.shape[
                                                                                                         0],
                                                                                                     sen1 / pred.shape[
                                                                                                         0],
                                                                                                     iou1 / pred.shape[
                                                                                                         0]))
            print('[{0:d}:{1:d}] --- loss:{2:.10f}\tacc2:{3:.4f}\tsen2:{4:.4f}\tiou2:{5:.4f}'.format(epoch + 1,
                                                                                                     iters, loss.item(),
                                                                                                     acc2 / pred.shape[
                                                                                                         0],
                                                                                                     sen2 / pred.shape[
                                                                                                         0],
                                                                                                     iou2 / pred.shape[
                                                                                                         0]))
            print('[{0:d}:{1:d}] --- loss:{2:.10f}\tacc3:{3:.4f}\tsen3:{4:.4f}\tiou3:{5:.4f}'.format(epoch + 1,
                                                                                                     iters, loss.item(),
                                                                                                     acc3 / pred.shape[
                                                                                                         0],
                                                                                                     sen3 / pred.shape[
                                                                                                         0],
                                                                                                     iou3 / pred.shape[
                                                                                                         0]))
            print('[{0:d}:{1:d}] --- loss:{2:.10f}\tacc4:{3:.4f}\tsen4:{4:.4f}\tiou4:{5:.4f}'.format(epoch + 1,
                                                                                                     iters, loss.item(),
                                                                                                     acc4 / pred.shape[
                                                                                                         0],
                                                                                                     sen4 / pred.shape[
                                                                                                         0],
                                                                                                     iou4 / pred.shape[
                                                                                                         0]))
            iters += 1
            # # ---------------------------------- visdom --------------------------------------------------
            X, x_acc, x_sen = iters, iters, iters
            Y, y_acc, y_sen = loss.item(), acc1 / pred.shape[0], sen1 / pred.shape[0]
            update_lines(env, panel, X, Y)
            update_lines(env1, panel1, x_acc, y_acc)
            update_lines(env2, panel2, x_sen, y_sen)
            # # --------------------------------------------------------------------------------------------

        adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)
        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, epoch + 1)

        # model eval
        if (epoch + 1) % args['test_step'] == 0:
            test_acc1, test_sen1, test_iou1, test_acc2, test_sen2, test_iou2, test_acc3, test_sen3, test_iou3, test_acc4, test_sen4, test_iou4 = model_eval(
                net)
            print("Average acc1:{0:.4f}, average sen1:{1:.4f}, average iou1:{2:.4f}".format(test_acc1, test_sen1,
                                                                                            test_iou1))
            print("Average acc2:{0:.4f}, average sen2:{1:.4f}, average iou2:{2:.4f}".format(test_acc2, test_sen2,
                                                                                            test_iou2))
            print("Average acc3:{0:.4f}, average sen3:{1:.4f}, average iou3:{2:.4f}".format(test_acc3, test_sen3,
                                                                                            test_iou3))
            print("Average acc4:{0:.4f}, average sen4:{1:.4f}, average iou4:{2:.4f}".format(test_acc4, test_sen4,
                                                                                            test_iou4))

            test_acc = np.mean([test_acc1, test_acc2, test_acc3, test_acc4])
            test_sen = np.mean([test_sen1, test_sen2, test_sen3, test_sen4])
            test_iou = np.mean([test_iou1, test_iou2, test_iou3, test_iou4])
            if (accuracy > test_acc) & (sensitivty > test_sen) & (IOU > test_iou):
                save_ckpt(net, epoch + 1 + 8888888)
                accuracy = test_acc
                sensitivty = test_sen
                IOU = test_iou


def model_eval(net):
    print("Start testing model...")
    test_data = Data(args['data_path'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)

    net.eval()
    Acc1, Sen1, IoU1 = [], [], []
    Acc2, Sen2, IoU2 = [], [], []
    Acc3, Sen3, IoU3 = [], [], []
    Acc4, Sen4, IoU4 = [], [], []

    file_num = 0
    for idx, batch in enumerate(batchs_data):
        image = batch[0].float().cuda()
        label = batch[1].float().cuda()
        pred_val = net(image)
        acc1, sen1, iou1, acc2, sen2, iou2, acc3, sen3, iou3, acc4, sen4, iou4 = metrics(pred_val, label,
                                                                                         pred_val.shape[0])
        print("\t---\t test acc1:{0:.4f}    test sen1:{1:.4f}    test iou1:{2:.4f}".format(acc1, sen1, iou1))
        print("\t---\t test acc2:{0:.4f}    test sen2:{1:.4f}    test iou2:{2:.4f}".format(acc2, sen2, iou2))
        print("\t---\t test acc3:{0:.4f}    test sen3:{1:.4f}    test iou3:{2:.4f}".format(acc3, sen3, iou3))
        print("\t---\t test acc4:{0:.4f}    test sen4:{1:.4f}    test iou4:{2:.4f}".format(acc4, sen4, iou4))
        Acc1.append(acc1)
        Sen1.append(sen1)
        IoU1.append(iou1)

        Acc2.append(acc2)
        Sen2.append(sen2)
        IoU2.append(iou2)

        Acc3.append(acc3)
        Sen3.append(sen3)
        IoU3.append(iou3)

        Acc4.append(acc4)
        Sen4.append(sen4)
        IoU4.append(iou4)

        file_num += 1
        # for better view, add testing visdom here.
        return np.mean(Acc1), np.mean(Sen1), np.mean(IoU1), np.mean(Acc2), np.mean(Sen2), np.mean(IoU2), np.mean(
            Acc3), np.mean(Sen3), np.mean(IoU3), np.mean(Acc4), np.mean(Sen4), np.mean(IoU4)


if __name__ == '__main__':
    train()
