from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np

def make_dataset(image_root, mask_root):
    imgs=[]
#    n=len(os.listdir(root))//2
    lists = os.listdir(image_root)
    for fileName in lists:
        img = os.path.join(image_root, fileName)
        #list2 = os.path.splitext(image_root + fileName)[0]
        mask = os.path.join(mask_root, fileName)
        imgs.append((img, mask))
    return imgs

#    for i in range(n):
 #       img=os.path.join(root,"%03d.png"%i)
#        mask=os.path.join(root,"%03d_mask.png"%i)
 #       imgs.append((img,mask))
 #   return imgs


class LiverDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None, target_transform=None):
        imgs = make_dataset(image_root, mask_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        # print(x_path)
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        # c
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        img_y = np.array(img_y)
        return img_x, img_y, y_path

    def __len__(self):
        return len(self.imgs)
