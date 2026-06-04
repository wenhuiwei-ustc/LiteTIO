import torch
import torch.utils.data as data
import numpy as np

import math
import random
from imageio import imread
from path import Path
import cv2
import warnings
warnings.filterwarnings("ignore", message=".*16-bit (uint16) PNG as int32.*")

def load_as_float(path):
    return np.expand_dims(imread(path).astype(np.float32), axis=2) # HW -> HW1

class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/Thermal/0000000.png
        root/scene_1/Thermal/0000001.png
        ..
        root/scene_1/cam.txt
        root/scene_2/Thermal/0000000.png
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """
    def __init__(self, root, seed=None, train=True, sequence_length=3,\
                 tf_share=None, tf_input=None, tf_loss=None, \
                 scene_type='indoor', interval=1):
        np.random.seed(seed)
        random.seed(seed)

        self.root = Path(root)
        if scene_type == 'indoor': 
            folder_list_path = self.root/'train_indoor.txt' if train else self.root/'val_indoor.txt'
        elif scene_type == 'outdoor': 
            folder_list_path = self.root/'train_outdoor.txt' if train else self.root/'val_outdoor.txt'
        elif scene_type == 'campus': 
            folder_list_path = self.root/'train_campus.txt' if train else self.root/'val_campus.txt'
        elif scene_type == 'sthe': 
            folder_list_path = self.root/'train_sthe.txt' if train else self.root/'val_sthe.txt'
        
        self.folders = [self.root/folder[:-1] for folder in open(folder_list_path)]
        self.tf_share = tf_share
        self.tf_input = tf_input
        self.tf_loss = tf_loss        
        self.crawl_folders(sequence_length, interval)

    def crawl_folders(self, sequence_length, interval):
        sequence_set = []
        demi_length = (sequence_length-1)//2 + interval - 1
        shifts = list(range(-demi_length, demi_length + 1))
        for i in range(1, 2*demi_length):
            shifts.pop(1)

        for folder in self.folders:
            imgs           = sorted((folder/"Thermal").files('*.png'))
            intrinsics     = np.genfromtxt(folder/'cam_T.txt').astype(np.float32).reshape((3, 3))
            imus           = sorted((folder/"IMU/data").files('*.txt'))

            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 'tgt_imu': imus[i], 'ref_imus': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                    sample['ref_imus'].append(imus[i+j])
                sequence_set.append(sample)

        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        sample = self.samples[index]

        # Read thermal images & GT depths
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        ## outdoor && indoor
        tgt_imu = np.genfromtxt(sample['tgt_imu']).astype(np.float32)[4:10]
        R_ic = np.array([[0.00265942, -0.99971627, -0.00516468], [0.00833786, -0.0053533, 0.99995321],
               [0.9999617, 0.02321018, 0.00817895]])
        tgt_imu1 = tgt_imu[:3]
        tgt_imu2 = tgt_imu[3:]
        tgt_imu1 = R_ic @ tgt_imu1
        tgt_imu2 = R_ic @ tgt_imu2
        tgt_imu = np.hstack((tgt_imu1, tgt_imu2))

        ref_imus = [np.hstack((R_ic @ np.genfromtxt(ref_imu).astype(np.float32)[4:7], R_ic @ np.genfromtxt(ref_imu).astype(np.float32)[7:10])) for ref_imu in sample['ref_imus']]
        
        # Pre-process thermal images for network input & loss calculation
        imgs, intrinsics = self.tf_share([tgt_img] + ref_imgs, np.expand_dims(np.copy(sample['intrinsics']),axis=0))
        imgs_input, _  = self.tf_input(imgs, None)
        imgs_loss, _   = self.tf_loss(imgs, None)

        tgt_img_input = imgs_input[0]
        ref_imgs_input = imgs_input[1:]
        tgt_img_loss = imgs_loss[0]
        ref_imgs_loss = imgs_loss[1:]
        tgt_imu_input = tgt_imu[:6]
        ref_imus_input = ref_imus[1:]

        return tgt_img_input, ref_imgs_input, tgt_img_loss, ref_imgs_loss, intrinsics.squeeze(), tgt_imu_input, ref_imus_input

    def __len__(self):
        return len(self.samples)
