import random
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform, user_transform
np.random.seed(0)


class ImageDataset(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        """Image generator

        Args:
            label_path (str): path to .csv file contains img paths and class labels
            cfg (str): path to .json file contains model configuration
            mode (str, optional): define which mode you are using. Defaults to 'train'.
            conditional_training (bool, optional): choosing train with conditional training or not. Defaults to False.
        """
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                        if self.dict[0].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path
                self._labels.append(labels)
                if flg_enhance and self._mode == 'train':
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = transform(image, self.cfg)
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))


class ImageDataset_full(Dataset):
    def __init__(self, label_path, cfg, mode='train', smooth_mode='pos', conditional_training=False):
        """Image generator for conditional training and finetuning parent samples

        Args:
            label_path (str): path to .csv file contains img paths and class labels
            cfg (str): path to .json file contains model configuration
            mode (str, optional): define which mode you are using. Defaults to 'train'.
            smooth_mode (str, optional): smoothing label regularization. Defaults to 'pos'.
            conditional_training (bool, optional): choosing train with conditional training or not. Defaults to False.
        """
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        if smooth_mode == 'pos':
            self.smooth_range = (0.55, 0.85)
        elif smooth_mode == 'neg':
            self.smooth_range = (0, 0.3)
        self.dict = {'1.0': 1.0, '': 0.0, '0.0': 0.0, '-1.0': -1.0}
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [header[7:]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]

                # check positive parent label
                positive_parent = ((smooth_mode == 'pos') and (self.dict.get(fields[5:][1]) != 0.0 or self.dict.get(fields[5:][3]) != 0.0)) \
                    or ((smooth_mode == 'neg') and (self.dict.get(fields[5:][1]) > 0.0 or self.dict.get(fields[5:][3]) > 0.0))

                # If not using conditional training => Load all images
                # Otherwise, only load images with parent label are positive or uncertain
                if mode == 'dev' or (not conditional_training or positive_parent):
                    for index, value in enumerate(fields[5:]):
                        labels.append(self.dict.get(value))
                    self._image_paths.append(image_path)
                    assert os.path.exists(image_path), image_path
                    self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = user_transform(image, self.cfg)
        labels = [random.uniform(self.smooth_range[0], self.smooth_range[1])
                  if x == -1.0 else x for x in self._labels[idx]]
        labels = np.array(labels).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
