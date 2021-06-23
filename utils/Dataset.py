import os
import cv2
import json
import torch
import torchvision
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

SUFFIX = ['.jpg', '.jpeg', '.png', '.bmp']

class FNDataset(Dataset):
    def __init__(self, root, img_size=128, transform=None, is_training=True):
        super(FNDataset, self).__init__()
        self.root = root
        self.img_size = img_size
        self.transform = transform
        self.is_training = is_training

        self.labels = []
        self.names = []
        for label in os.listdir(os.path.join(self.root, 'anno')):
            for name in os.listdir(os.path.join(self.root, 'anno', label)):
                name = name[:-5] + '.jpg'
                if os.path.splitext(name)[-1] not in SUFFIX:
                    continue
                self.labels.append(label)
                self.names.append(name)
        self.labels = np.array(self.labels)
        self.names = np.array(self.names)

    def __len__(self):
        return len(self.names)

    def _crop_img(self, img, bbox, lndm, ratio=0.8):
        W, H = img.size
        w, h = bbox[2:]
        size = int((w + h) * ratio)
        bbox[0] = 0 if bbox[0] < 0 else bbox[0]
        bbox[1] = 0 if bbox[1] < 0 else bbox[1]
        center_x = bbox[0] + int(w / 2)
        center_y = bbox[1] + int(h * 3 / 7)
        size = W if size > W else size
        size = H if size > H else size

        if center_x - int(size / 2) < 0:
            left = 0
            right = left + size
        else:
            left = center_x - int(size / 2)
            right = left + size
        if right > W:
            right = W
            left = right - size

        if center_y - int(size / 2) < 0:
            top = 0
            bottom = top + size
        else:
            top = center_y - int(size / 2)
            bottom = top + size
        if bottom > H:
            bottom = H
            top = bottom - size
        bbox = [left, top, right, bottom]
        crop_img = img.crop(bbox)
        lndm = (np.array(lndm) - np.array([left, top])) / size
        lndm = np.clip(lndm, a_min=0., a_max=1.)

        if self.is_training and random.random() < 0.5:
            crop_img = crop_img.transpose(Image.FLIP_LEFT_RIGHT)
            lndm[:, 0] = 1 - lndm[:, 0]
            bbox[0] = size - bbox[0]
            bbox[2] = size - bbox[2]

        return crop_img, bbox, lndm

    def __getitem__(self, idx):
        label = self.labels[idx]
        name = self.names[idx]
        clr_path = os.path.join(self.root, 'clr', label, name)
        anno_path = os.path.join(self.root, 'anno', label, name.split('.')[0] + '.json')
        with open(anno_path, 'r') as inner:
            anno = json.load(inner)
        bbox = anno['bbox']
        lndm = anno['lndm']
        img = Image.open(clr_path)
        crop_img, bbox, lndm = self._crop_img(img, bbox, lndm)

        crop_img = crop_img.resize((self.img_size, self.img_size), Image.LANCZOS)
        msk = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        outlines = lndm[:33]*np.array([self.img_size, self.img_size])
        cv2.fillConvexPoly(msk, outlines.astype('int'), (255, 255, 255))

        crop_img = self.transform(crop_img)
        msk = transforms.ToTensor()(msk)
        msk = (msk < 0.5).float()
        lndm = torch.as_tensor(lndm, dtype=torch.float32)

        if not self.is_training:
            return crop_img, msk, lndm

        if random.random() < 0.5:
            same_label_idx = random.choice(np.where(label == self.labels)[0])
            label_idx = same_label_idx
        else:
            diff_label_idx = random.choice(np.arange(len(self.labels)))
            label_idx = diff_label_idx
        ide_label = self.labels[label_idx]
        ide_name = self.names[label_idx]
        ide_clr_path = os.path.join(self.root, 'clr', ide_label, ide_name)
        ide_anno_path = os.path.join(self.root, 'anno', ide_label, ide_name.split('.')[0] + '.json')
        with open(ide_anno_path, 'r') as inner:
            ide_anno = json.load(inner)
        ide_bbox = ide_anno['bbox']
        ide_lndm = ide_anno['lndm']
        ide_img = Image.open(ide_clr_path)
        ide_crop_img, _, _ = self._crop_img(ide_img, ide_bbox, ide_lndm)
        ide_crop_img = ide_crop_img.resize((self.img_size, self.img_size), Image.LANCZOS)
        ide_crop_img = transforms.ToTensor()(ide_crop_img)
        # ide_crop_img = transforms.Grayscale()(ide_crop_img)

        return crop_img, ide_crop_img, msk, lndm #, label, name


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataset = FNDataset('../dataset', transform=transform, is_training=True)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, num_workers=10, shuffle=True, drop_last=True)
    from tqdm import tqdm
    for data in tqdm(loader):
        img, iden_img, msk, lndm = data
