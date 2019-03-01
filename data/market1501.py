from data.common import list_pictures

from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader

import pandas as pd
import numpy as np

class Market1501(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader

        data_path = args.datadir

        if dtype == 'train':
            data_path += 'pre_train'
            self.is_train = True
        elif dtype == 'test':
            data_path += 'new_test'
            self.is_train = False
        else:
            data_path += 'query'
            # data_path += 'new_train'
            self.is_train = True

        if  self.is_train == False:
            self.submission = pd.read_csv(args.datadir + '/sample_submission.csv')['Image'].values.tolist()
            self.imgs = [data_path + '/' + path for path in self.submission]  # 图首不为-1

        if self.is_train:
            self.imgs = [data_path + '/' + path for path in list_pictures(data_path)]  # 图首不为-1
            self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        if self.is_train :
            target = self._id2label[self.id(path)]
        else:
            target = self.submission[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):                                               #图片名与id对应,主要改这个函数
        """
        :param file_path: unix style file path
        :return: person id
        """
        #return int(file_path.split('/')[-1].split('_')[0])
        return file_path.split('/')[-1].split('.')[1]


    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]
