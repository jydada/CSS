import torch
import argparse
import random
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torchvision.transforms import transforms
from dataset import LiverDataset
import scipy.io
import numpy as np
import warnings
from PIL import Image
import skimage.io as io
from skimage import data_dir
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from utils import *
from Models import AttU_Net
from utils.dice_score import dice_loss
#1. set random.seed and cudnn performance
random.seed(888)
np.random.seed(888)
torch.manual_seed(888)
torch.cuda.manual_seed_all(888)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


nums_class = 2

x_transforms = transforms.Compose([
    # transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.7956, 0.7461, 0.8198], [0.1309, 0.1553, 0.1064])
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    # transforms.Resize((512,512)),
    #transforms.ToTensor(),
])


def train_model(model, criterion, optimizer, train_dataload, val_dataload, num_epochs=100):
    metric1 = 0
    metric2 = 0
    for epoch in range(num_epochs):

        train_progressor = ProgressBar(mode="Train", epoch=epoch, total_epoch=num_epochs,
                                       model_name="Unet", total=len(train_dataload.dataset))
        dt_size = len(train_dataload.dataset)
        epoch_loss = 0
        step = 0
        train_losses = AverageMeter()
        for iter,(x, y, _) in enumerate(train_dataload):
            train_progressor.current = iter
            step += 1
            inputs = x.cuda()
            labels = y.cuda().long()
            #print(y_path)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            # pred = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            # loss2 = dice_loss(pred, labels)
            loss2 = dice_loss(F.softmax(outputs, dim=1).float(),
                              F.one_hot(labels, 2).permute(0, 3, 1, 2).float(),
                              multiclass=True)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(loss.item())
            # print(inputs.size(0))
            train_losses.update(loss.item(), inputs.size(0))
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = 0.45
            train_progressor()
        train_progressor.done()
        # print("epoch %d %d/%d,train_loss:%0.3f" % (epoch, step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        # print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
        val_mIOU, val_mdice = validation(model, val_dataload, nums_class, val_dataload.dataset.__len__(), meanIOU=True)
        # print('Validation mIoU:{:.6f} mDICE:{:.6f}'.format(val_mIOU, val_mdice))
        if val_mIOU > metric1 or val_mdice > metric2:
            metric1 = val_mIOU
            metric2 = val_mdice
            torch.save(model.state_dict(), 'AttU_Net_MTSTnuclei.pth')
            print('Validation mIoU:{:.6f} mDICE:{:.6f}'.format(val_mIOU, val_mdice))
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), str(epoch + 1) + 'AttU_Net_MTSTnuclei.pth')
    return model

#训练模型
def train(args):
    batch_size = args.batch_size
    train_dataset = LiverDataset(".\\MTST\\train\\images\\", ".\\MTST\\train\\masks\\", transform=x_transforms, target_transform= None)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_dataset = LiverDataset(".\\MTST\\val\\images\\", ".\\MTST\\val\\masks\\", transform=x_transforms,
                                 target_transform=None)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    model = AttU_Net(3, 2)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, amsgrad=True, weight_decay=1e-5)

    train_model(model, criterion, optimizer, train_dataloader,val_dataloader)

    # val_mIOU, val_mdice = validation(model, dataloaders, nums_class, liver_dataset.__len__(), meanIOU=True)
    # print('Validation mIoU:{:.6f} mDICE:{:.6f}'.format(val_mIOU, val_mdice))
    # if val_mIOU > metric:
    #     metric = val_mIOU
    #     torch.save(model.state_dict(), 'unet_%d.pth' % (int(metric * 10000)))


def cal_iou(output, target, K, allNums=2560, mean_iou= True):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    # assert (output.dim() in [1, 2, 3])
    # assert output.shape == target.shape
    output = output.view(output.shape[0], -1)
    target = target.view(target.shape[0], -1)

    # # average the batch_size
    for ind in range(output.shape[0]):
        one_output = output[ind, :]
        one_target = target[ind, :]
        # one_output[one_target == ignore_index] = ignore_index
        intersection = one_output[one_output == one_target]
        area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1) # get n_{ii} in confuse matrix
        area_target = torch.histc(one_target, bins=K, min=0, max=K-1)
        area_output = torch.histc(one_output, bins=K, min=0, max=K-1)
        area_union = area_output + area_target - area_intersection
        one_iou_class = area_intersection.float() / (area_union.float() + 1e-10)
        one_dice_class = 2*area_intersection.float() / (area_union.float() + area_intersection.float() + 1e-10)

        if not ind:
            iou_class = one_iou_class
            dice_class = one_dice_class
        else:
            iou_class += one_iou_class
            dice_class += one_dice_class

    iou_class = iou_class / allNums
    dice_class = dice_class / allNums

    if mean_iou:
        mIoU = torch.mean(iou_class)
        mdice = torch.mean(dice_class)
        return mIoU.cpu().numpy(), mdice.cpu().numpy()
    else:
        # weightes = (np.argsort(np.argsort(iou_class.cpu().numpy())[::-1]) / (K-1)) + 0.0001
        # weights = torch.tensor(weightes).float().cuda()
        return iou_class.cpu().numpy(), dice_class.cpu().numpy()

def validation(network, loader, class_nums, sample_nums, meanIOU = True):
    if meanIOU:
        mIOUs = 0.0
        mdices = 0.0
    else:
        mIOUs = np.zeros((class_nums))
        mdices = np.zeros((class_nums))

    network.eval()
    with torch.no_grad():
        for i, (img, target, _) in tqdm(enumerate(loader)):
            img = img.cuda()
            target = target.cuda().long()

            outputs = network(img)
            prediction = torch.argmax(F.softmax(outputs, 1, _stacklevel=5), dim=1)

            m, d = cal_iou(prediction, target, class_nums, sample_nums, mean_iou =meanIOU)
            mIOUs += m
            mdices += d
    return mIOUs, mdices

if __name__ == '__main__':
    #参数解析
    #parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, default='train', help="train or test")
    parse.add_argument("--batch_size", type=int, default=5)
    parse.add_argument("--ckpt", type=str, default='Unet_nuclei.pth', help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
