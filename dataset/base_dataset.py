# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys
sys.path.append('.')
import os
import yaml
import glob
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict
import argparse
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from dataset.albu import IsotropicResize

class DeepfakeAbstractBaseDataset(data.Dataset):
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
        self.label_list = []

        if mode == 'train':
            dataset_list = config['train_dataset']
        elif mode == 'test':
            dataset_list = config['test_dataset']
        else:
            raise NotImplementedError('Only train and test modes are supported.')
        self.dataset_list = dataset_list

        image_list, domain_label_list = self.collect_img_and_label(dataset_list)
        self.image_list, self.domain_label_list = image_list, domain_label_list
        self.label_list = [1 if d != 0 else 0 for d in self.domain_label_list]

        self.data_dict = {
            'image': self.image_list,
            'domain_label': self.domain_label_list,
            'label': self.label_list,
        }

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
        ], 
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def collect_img_and_label(self, dataset_list):
        label_list = []
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
                    if cp is None and dataset_name.split('_')[0] in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                        sub_dataset_info = sub_dataset_info[self.compression]
                    elif cp == 'c40':
                        sub_dataset_info = sub_dataset_info['c40']

                    for video_name, video_info in sub_dataset_info.items():
                        if video_info['label'] not in self.config['label_dict']:
                            raise ValueError(f"Label {video_info['label']} is not found in the configuration file.")
                        domain_label = self.config['label_dict'][video_info['label']]
                        frame_paths = video_info['frames']
                        label_list.extend([domain_label] * len(frame_paths))
                        frame_path_list.extend(frame_paths)

                if self.mode == 'test':
                    seed = 42
                    rnd = random.Random(seed)
                    shuffled = list(zip(label_list, frame_path_list))
                    rnd.shuffle(shuffled)
                    label_list, frame_path_list = zip(*shuffled)

                return frame_path_list, label_list
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
            mask = cv2.resize(mask, (size, size))/255
            mask = np.expand_dims(mask, axis=2)
            return np.float32(mask)
        else:
            return np.zeros((size, size, 1))

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if os.path.exists(file_path):
            landmark = np.load(file_path)
            return np.float32(landmark)
        else:
            return np.zeros((81, 2))

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

    def data_aug(self, img, landmark=None, mask=None):
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
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask')

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index):
        image_path = self.data_dict['image'][index]
        domain_label = self.data_dict['domain_label'][index]
        label = self.data_dict['label'][index]
        mask_path = image_path.replace('frames', 'masks')
        landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')

        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)

        mask = self.load_mask(mask_path) if self.config['with_mask'] else None
        landmarks = self.load_landmark(landmark_path) if self.config['with_landmark'] else None

        if self.config['use_data_augmentation']:
            image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask)
        else:
            image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

        image_trans = self.normalize(self.to_tensor(image_trans))
        if self.config['with_landmark']:
            landmarks_trans = torch.from_numpy(landmarks)
        if self.config['with_mask']:
            mask_trans = torch.from_numpy(mask_trans)

        return image_trans, domain_label, label, landmarks_trans, mask_trans, image_path

    @staticmethod
    def collate_fn(batch):
        image, domain_label, label, landmarks, mask, image_path = zip(*batch)
        image = torch.stack(image, dim=0)
        domain_label = torch.LongTensor(domain_label)
        label = torch.LongTensor(label)
        image_path = list(image_path)

        landmarks = torch.stack(landmarks, dim=0) if landmarks[0] is not None else None
        mask = torch.stack(mask, dim=0) if mask[0] is not None else None

        return {
            'image': image,
            'domain_label': domain_label,
            'label': label,
            'landmark': landmarks,
            'mask': mask,
            'image_path': image_path,
        }

    def __len__(self):
        assert len(self.image_list) == len(self.domain_label_list), 'Number of images and domain_labels are not equal'
        return len(self.image_list)


if __name__ == "__main__":
    with open('dataset_config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='train',
                # ae = True
            )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, data_dict in enumerate(tqdm(train_data_loader)):
        image_path = \
            data_dict['image_path']
        print(image_path[0])
        print(type(image_path[0]))
        image_path[0].split('\\')[-3]