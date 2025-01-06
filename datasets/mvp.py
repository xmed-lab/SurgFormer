from pathlib import Path

import torch
from torch.utils.data import Dataset
# import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random
import pickle
from torchvision import models, transforms
import torchvision.transforms.functional as TF

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)

        return img_

class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees, self.degrees)
        return TF.rotate(img, angle)

class MVPDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform, train_num_each_28, sequence_length, loader=pil_loader):
        self.file_paths = file_paths  # all the samples 208182
        self.file_labels_phase = file_labels[:, 0]   # all the samples 208182
        self.file_labels_tool = file_labels[:, 1: 9]
        self.file_labels_phase_ant = file_labels[:, 9: 21]
        self.file_labels_tool_ant = file_labels[:, 21: 29]
        self.transform = transform
        self.loader = loader
        self.train_num_each = train_num_each_28
        self.sequence_length = sequence_length
        self.phase_classes = 12

        self.prepare_metas(train_num_each_28, sequence_length)
        # print('initial finished')

    def prepare_metas(self, list_each_length, sequence_length):
        self.metas = []
        interval = int(sequence_length/2)
        count = 0
        for i in range(len(list_each_length)):
            for j in range(count, count + (list_each_length[i] + 1 - sequence_length), interval):
                sequece_idx = [k for k in range(j, j+sequence_length)]
                # idx.append(j)
                self.metas.append(sequece_idx)
            count += list_each_length[i]


    def __getitem__(self, index):
        seque_idx = self.metas[index]
        shift_num = random.randint(0, 10)
        if seque_idx[-1] + shift_num + 1 <= len(self.file_paths):
            start = seque_idx[0] + shift_num
            end = seque_idx[-1] + shift_num + 1
            seque_idx_new = [i for i in range(start, end)]
            seque_idx = seque_idx_new
        else:
            seque_idx = seque_idx
        imgs, labels_phases, labels_tools, labels_phase_ants, labels_tool_ants = [], [], [], [], []
        for inx in seque_idx:
            img_name = self.file_paths[inx]
            img = self.loader(img_name)
            img = self.transform(img)
            imgs.append(img)

            labels_phase = int(self.file_labels_phase[inx])
            labels_phase_onehot = torch.zeros(self.phase_classes)
            labels_phase_onehot[labels_phase] = 1
            labels_phases.append(labels_phase_onehot)

            labels_tool = self.file_labels_tool[inx]
            labels_tool = torch.from_numpy(labels_tool)
            labels_tools.append(labels_tool)

            labels_phase_ant = self.file_labels_phase_ant[inx]
            labels_phase_ant = torch.from_numpy(labels_phase_ant)
            labels_phase_ants.append(labels_phase_ant)

            labels_tool_ant = self.file_labels_tool_ant[inx]
            labels_tool_ant = torch.from_numpy(labels_tool_ant)
            labels_tool_ants.append(labels_tool_ant)


        imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
        labels_phases = torch.stack(labels_phases, dim=0)
        labels_tools = torch.stack(labels_tools, dim=0)
        labels_phase_ants = torch.stack(labels_phase_ants, dim=0)
        labels_tool_ants = torch.stack(labels_tool_ants, dim=0)

        target = {
            'labels_phase': labels_phases,  # Tx12
            'labels_tool': labels_tools,    # Tx8
            'labels_phase_ant': labels_phase_ants,  #Tx12
            'labels_tool_ant': labels_tool_ants  # Tx8
        }
        return imgs, target

    def __len__(self):
        return len(self.metas)

def build(image_set, args):
    PATHS = {
        "train": ('/nfs/usrhome/eemenglan/71_heart/train_paths_labels1_4task_30.pkl'),
        "val": ('/nfs/usrhome/eemenglan/71_heart/test_paths_labels1_4task_30.pkl'),  # not used actually
    }
    data_path = PATHS[image_set]
    with open(data_path, 'rb') as f:
        train_paths_labels = pickle.load(f)
    train_paths_30 = train_paths_labels[0]
    train_labels_30 = train_paths_labels[1]
    train_num_each_30 = train_paths_labels[2]

    # print('train_paths_28  : {:6d}'.format(len(train_paths_28)))
    # print('train_labels_28 : {:6d}'.format(len(train_labels_28)))


    train_labels_30 = np.asarray(train_labels_30, dtype=np.float32)

    train_transforms = None
    # test_transforms = None

    train_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        # RandomCrop(224),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        # RandomHorizontalFlip(),
        # RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # sequence_length = 32
    # train_dataset_19 = CholecDataset(train_paths_19, train_labels_19, train_transforms)
    train_dataset_30 = MVPDataset(train_paths_30, train_labels_30, train_transforms, train_num_each_30, args.sequence_length)


    return train_dataset_30 #, train_num_each_28

if __name__ == '__main__':
    train_dataset = build('train')
    imgs, target = train_dataset.__getitem__(10)
    # sum_num = sum(train_num_each)
    # num_train = len(train_dataset)
    # sequence_length = 100
    # train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    # num_train_we_use_80 = len(train_useful_start_idx)
    # train_idx_80 = []
    # for i in range(num_train_we_use_80):
    #     for j in range(sequence_length):
    #         train_idx_80.append(train_useful_start_idx[i] + j)

    print('test done')
