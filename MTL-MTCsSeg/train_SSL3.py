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
# random.seed(888)
# np.random.seed(888)
# torch.manual_seed(888)
# torch.cuda.manual_seed_all(888)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.backends.cudnn.benchmark = True
# warnings.filterwarnings('ignore')

args = {
    'root'      : '',
    'data_path' : 'dataset/DRIVE7/',
    'pretext_data_path' : 'dataset/EPST/',
    'pretext_data2_path' : 'dataset/DRIVE1/',
    'epochs'    : 100,
    'lr'        : 0.0001,
    'snapshot'  : 10,
    'test_step' : 1,
    'ckpt_path' : 'checkpoint-3/100/',
    'batch_size': 5,
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

    pretext_data1 = Data(args['pretext_data_path'], train=True)
    pretext_batchs_data1 = DataLoader(pretext_data1, batch_size=4, num_workers=2, shuffle=True)
    pretext_data2 = Data(args['pretext_data2_path'], train=True)
    pretext_batchs_data2 = DataLoader(pretext_data2, batch_size=4, num_workers=2, shuffle=True)
    print(pretext_data2.__len__()//train_data.__len__())
    ii=pretext_data2.__len__() // train_data.__len__()
    train_data = []
    for i in range(ii*5//4):
        print(i)
        train_data1 = Data(args['data_path'], train=True)
        train_data.extend(train_data1)
    batchs_data = DataLoader(train_data, batch_size=5, num_workers=2, shuffle=True)

    iters = 1
    accuracy = 0.
    sensitivty = 0.
    IOU = 0.
    for epoch in range(args['epochs']):
        net.train()
        for idx, (batch, batch2, batch3) in enumerate(zip(batchs_data, pretext_batchs_data1, pretext_batchs_data2)):
            image = batch[0].cuda()
            label = batch[1].cuda().long()
            image2 = batch2[0].cuda()
            label2 = batch2[1].cuda().long()
            image3 = batch3[0].cuda()
            label3 = batch3[1].cuda().long()
            optimizer.zero_grad()
            pred = net(image)
            pred2 = net(image2)
            pred3 = net(image3)
            # pred = pred.squeeze_(1)
            #pred = torch.argmax(pred, dim=1)
            #label = label.squeeze(1)
            loss = critrion(pred, label)
            loss2 = critrion(pred2, label2)
            loss3 = critrion(pred3, label3)
            #loss2 = dice_coeff_loss(pred, label)
            loss = loss + loss2*0.25 + loss3*0.75
            loss.backward()
            optimizer.step()
            acc, sen, iou = metrics(pred, label, pred.shape[0])
            print('[{0:d}:{1:d}] --- loss:{2:.10f}\tacc:{3:.4f}\tsen:{4:.4f}\tiou:{5:.4f}'.format(epoch + 1,
                                                                                     iters, loss.item(),
                                                                                     acc / pred.shape[0],
                                                                                     sen / pred.shape[0],
                                                                                     iou / pred.shape[0]))
            iters += 1
            # # ---------------------------------- visdom --------------------------------------------------
            X, x_acc, x_sen = iters, iters, iters
            Y, y_acc, y_sen = loss.item(), acc / pred.shape[0], sen / pred.shape[0]
            update_lines(env, panel, X, Y)
            update_lines(env1, panel1, x_acc, y_acc)
            update_lines(env2, panel2, x_sen, y_sen)
            # # --------------------------------------------------------------------------------------------

        adjust_lr(optimizer, base_lr=args['lr'], iter=epoch, max_iter=args['epochs'], power=0.9)
        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, epoch + 1)

        # model eval
        if (epoch + 1) % args['test_step'] == 0:
            test_acc, test_sen , test_iou = model_eval(net)
            print("Average acc:{0:.4f}, average sen:{1:.4f}, average iou:{2:.4f}".format(test_acc, test_sen, test_iou))

            if (accuracy > test_acc) & (sensitivty > test_sen) & (IOU >test_iou) :
                save_ckpt(net, epoch + 1 + 8888888)
                accuracy = test_acc
                sensitivty = test_sen
                IOU = test_iou


def model_eval(net):
    print("Start testing model...")
    test_data = Data(args['data_path'], train=False)
    batchs_data = DataLoader(test_data, batch_size=1)

    net.eval()
    Acc, Sen, IoU = [], [], []
    file_num = 0
    for idx, batch in enumerate(batchs_data):
        image = batch[0].float().cuda()
        label = batch[1].float().cuda()
        pred_val = net(image)
        acc, sen ,iou = metrics(pred_val, label, pred_val.shape[0])
        print("\t---\t test acc:{0:.4f}    test sen:{1:.4f}    test iou:{2:.4f}".format(acc, sen, iou))
        Acc.append(acc)
        Sen.append(sen)
        IoU.append(iou)
        file_num += 1
        # for better view, add testing visdom here.
        return np.mean(Acc), np.mean(Sen), np.mean(IoU)


if __name__ == '__main__':
    train()
