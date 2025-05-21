# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys
sys.path.append('.')

import os
import yaml
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from dataset.albu import IsotropicResize

class DFPDDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """

    def __init__(self, config=None, mode='train'):
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]
        self.image_list = []
        self.domain_label_list = []

        if mode == 'train':
            dataset_list = config['train_dataset']
        elif mode == 'test':
            dataset_list = config['test_dataset']
        else:
            raise NotImplementedError('Only train and test modes are supported.')
        self.dataset_list = dataset_list

        image_list, domain_label_list = self.collect_img_and_label(dataset_list)
        self.image_list, self.domain_label_list = image_list, domain_label_list

        self.data_dict = {
            'image': self.image_list,
            'domain_label': self.domain_label_list,
        }

        # 仅保留 domain_label == 0 的索引，用于主图像采样
        self.valid_main_indices = [i for i, label in enumerate(self.domain_label_list) if label == 0]
        # 仅保留 domain_label != 0 的索引，用于 paired 图像采样
        self.fake_indices = [i for i, label in enumerate(self.domain_label_list) if label != 0]

        self.transform = self.init_data_aug_method()

    def init_data_aug_method(self):
        trans = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ])
        return trans

    def collect_img_and_label(self, dataset_list):
        domain_label_list = []
        frame_path_list = []

        if dataset_list:
            for dataset_name in dataset_list:
                with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                    dataset_info = json.load(f)
                cp = None
                if dataset_name.endswith('_c40'):
                    cp = 'c40'

                for label in dataset_info[dataset_name]:
                    sub_dataset_info = dataset_info[dataset_name][label][self.mode]
                    if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                        sub_dataset_info = sub_dataset_info[self.compression]
                    elif cp == 'c40' and dataset_name in ['FF-DF_c40', 'FF-F2F_c40', 'FF-FS_c40', 'FF-NT_c40', 'FaceForensics++_c40','FaceShifter_c40']:
                        sub_dataset_info = sub_dataset_info['c40']
                    for video_name, video_info in sub_dataset_info.items():
                        if video_info['label'] not in self.config['label_dict']:
                            raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                        domain_label = self.config['label_dict'][video_info['label']]
                        frame_paths = video_info['frames']

                        domain_label_list.extend([domain_label]*len(frame_paths))
                        frame_path_list.extend(frame_paths)

                shuffled = list(zip(domain_label_list, frame_path_list))
                random.shuffle(shuffled)
                domain_label_list, frame_path_list = zip(*shuffled)

                return frame_path_list, domain_label_list

        else:
            raise ValueError('No dataset is given.')

    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution']
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if os.path.exists(file_path):
            mask = cv2.imread(file_path, 0)
            if mask is None:
                mask = np.zeros((size, size))
            mask = cv2.resize(mask, (size, size)) / 255
            mask = np.expand_dims(mask, axis=2)
            return np.float32(mask)
        else:
            return np.zeros((size, size, 1))

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, mask=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Create a dictionary of arguments
        kwargs = {'image': img}

        # Check if the landmark and mask are not None
        if mask is not None:
            kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_mask = transformed.get('mask')

        return augmented_img, augmented_mask

    def __getitem__(self, idx):
        index = self.valid_main_indices[idx]  # 只使用 label=0 的主图像
        image_path = self.data_dict['image'][index]
        domain_label = self.data_dict['domain_label'][index]

        paired_index = self.get_paired_index(index)
        paired_image_path = self.data_dict['image'][paired_index]
        paired_domain_label = self.data_dict['domain_label'][paired_index]

        mask_path = image_path.replace('frames', 'masks')
        paired_mask_path = paired_image_path.replace('frames', 'masks')

        try:
            image = self.load_rgb(image_path)
            paired_image = self.load_rgb(paired_image_path)
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)

        mask = self.load_mask(mask_path) if self.config['with_mask'] else None
        paired_mask = self.load_mask(paired_mask_path) if self.config['with_mask'] else None

        if self.config['use_data_augmentation']:
            image_trans, mask_trans = self.data_aug(image, mask)
            paired_image_trans, paired_mask_trans = self.data_aug(paired_image, paired_mask)
        else:
            image_trans, paired_image_trans, _, mask_trans = deepcopy(image), deepcopy(paired_image), None, deepcopy(
                mask)
            paired_mask_trans = deepcopy(paired_mask)

        image_trans = self.normalize(self.to_tensor(image_trans))
        paired_image_trans = self.normalize(self.to_tensor(paired_image_trans))
        if self.config['with_mask']:
            mask_trans = torch.from_numpy(mask_trans)
            paired_mask_trans = torch.from_numpy(paired_mask_trans)

        label = 1 if domain_label != 0 else 0
        paired_label = 1 if paired_domain_label != 0 else 0

        return image_trans, paired_image_trans, domain_label, paired_domain_label, label, paired_label, mask_trans, paired_mask_trans

    @staticmethod
    def collate_fn(batch):
        image, paired_image, domain_label, paired_domain_label, label, paired_label, mask, paired_mask = zip(*batch)

        image = torch.stack(image + paired_image, dim=0)
        domain_label = torch.LongTensor(domain_label + paired_domain_label)
        label = torch.LongTensor(label + paired_label)
        if mask[0] is not None:
            mask = torch.stack(mask + paired_mask, dim=0)
        else:
            mask = None

        return {
            'image': image,
            'domain_label': domain_label,
            'label': label,
            'mask': mask,
        }

    def __len__(self):
        return len(self.valid_main_indices)

    def get_paired_index(self, index):
        """
        Return an index of a sample whose domain_label != 0,
        ensuring paired_label == 1 while main image label == 0.
        """
        return random.choice(self.fake_indices)
