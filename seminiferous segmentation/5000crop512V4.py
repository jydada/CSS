import PIL.Image as Image

Image.MAX_IMAGE_PIXELS = None
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F
import warnings
import random
from torch import nn, optim
import cv2
from Models import AttU_Net
random.seed(888)
np.random.seed(888)
torch.manual_seed(888)
torch.cuda.manual_seed_all(888)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

x_transforms = transforms.Compose([
    # transforms.Resize((1200,1200)),
    transforms.ToTensor(),
    transforms.Normalize([0.7956, 0.7461, 0.8198], [0.1309, 0.1553, 0.1064])
])

model = AttU_Net(3, 2).cuda()
model.load_state_dict(torch.load('AttU_Net_MTSTnuclei.pth'))
model.eval()

file_root = "./1/"

save_root = "./out/"

patch_size = 512

files = os.listdir(file_root)
for Name in files:
    # print(Name)
    if not os.path.exists(save_root + Name):
        os.makedirs(save_root + Name)
    image_root = file_root + Name
    lists = os.listdir(image_root)
    # print(lists)
    step = 0
    for fileName in lists:
        step += 1
        img_path = os.path.join(image_root, fileName)
        fileName1 = os.path.splitext(fileName)[0]
        # print(len(lists))
        print(" %s  %d / %d" % (fileName1, step, len(lists)))
        # print(fileName1)
        list2 = os.path.splitext(img_path)[0]
        # print(list2)
        img = Image.open(img_path)
        img = np.array(img)
        height, width, channel = img.shape
        # print(width)
        t = math.floor(width / patch_size)
        s = math.floor(height / patch_size)
        # print(s)
        b = np.zeros((height, width), dtype=float)
        for i in range(s + 1):
            # print(i)
            if patch_size * i + patch_size <= height:
                for j in range(t + 1):
                    if patch_size * j + patch_size <= width:
                        a = img[patch_size * i:patch_size * i + patch_size, patch_size * j:patch_size * j + patch_size,
                            :]
                        tensor_a = x_transforms(a)
                        tensor_a = tensor_a.unsqueeze(0)
                        inputs = tensor_a.cuda()
                        y = model(inputs)
                        img_y = torch.argmax(F.softmax(y, dim=1), dim=1)
                        img_y = img_y.squeeze(0).cpu().numpy().astype(np.uint8)
                        b[patch_size * i:patch_size * i + patch_size,
                        patch_size * j:patch_size * j + patch_size] = img_y
                        # im = Image.fromarray(img_y)
                        # im.save(save_root + fileName1 + '_' + str(i) + '_' + str(j) + '.tif')
                    elif patch_size * j < width:
                        a = img[patch_size * i:patch_size * i + patch_size, patch_size * j:width, :]
                        a = cv2.resize(a, (patch_size, patch_size))
                        tensor_a = x_transforms(a)
                        tensor_a = tensor_a.unsqueeze(0)
                        inputs = tensor_a.cuda()
                        y = model(inputs)
                        img_y = torch.argmax(F.softmax(y, dim=1), dim=1)
                        img_y = img_y.squeeze(0).cpu().numpy().astype(np.uint8)
                        img_y = cv2.resize(img_y, (width - j * patch_size, patch_size))
                        b[patch_size * i:patch_size * i + patch_size, patch_size * j:width] = img_y
                        # im = Image.fromarray(img_y)
                        # im.save(save_root + fileName1 + '_' + str(i) + '_' + str(j) + '.tif')
            elif patch_size * i < height:
                for j in range(s + 1):
                    if patch_size * j + patch_size <= width:
                        a = img[patch_size * i:height, patch_size * j:patch_size * j + patch_size, :]
                        a = cv2.resize(a, (patch_size, patch_size))
                        tensor_a = x_transforms(a)
                        tensor_a = tensor_a.unsqueeze(0)
                        inputs = tensor_a.cuda()
                        y = model(inputs)
                        img_y = torch.argmax(F.softmax(y, dim=1), dim=1)
                        img_y = img_y.squeeze(0).cpu().numpy().astype(np.uint8)
                        img_y = cv2.resize(img_y, (patch_size, height - i * patch_size))
                        b[patch_size * i:height, patch_size * j:patch_size * j + patch_size] = img_y
                        # im = Image.fromarray(img_y)
                        # im.save(save_root + fileName1 + '_' + str(i) + '_' + str(j) + '.tif')
                    elif patch_size * j < width:
                        a = img[patch_size * i:height, patch_size * j:width, :]
                        a = cv2.resize(a, (patch_size, patch_size))
                        tensor_a = x_transforms(a)
                        tensor_a = tensor_a.unsqueeze(0)
                        inputs = tensor_a.cuda()
                        y = model(inputs)
                        img_y = torch.argmax(F.softmax(y, dim=1), dim=1)
                        img_y = img_y.squeeze(0).cpu().numpy().astype(np.uint8)
                        img_y = cv2.resize(img_y, (width - j * patch_size, height - i * patch_size))
                        b[patch_size * i:height, patch_size * j:width] = img_y
                        # im = Image.fromarray(img_y)
                        # im.save(save_root + fileName1 + '_' + str(i) + '_' + str(j) + '.tif')
        # plt.imshow(b)
        # plt.show()
        im = Image.fromarray(b)
        im.save(save_root + Name + '\\' + fileName1 + '.tif')
    # list2 = os.path.splitext(image_root + fileName)[0]
    # mask = os.path.join(mask_root, fileName)
    # imgs.append((img, mask))
