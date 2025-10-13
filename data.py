import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.transform import radon, iradon
import utils


class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, saved_path, test_patient, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0, 1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))   # glob遍历文件夹下所有文件或文件夹；sorted对所有可迭代的对象进行排序操作
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        self.mode = mode
        self.load_mode = load_mode
        self.transform = transform

        if mode == 'train':
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
            else:  # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
        else:  # mode =='test'
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
            else:    # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)
            # theta0 = np.linspace(0., 180., num=720, endpoint=False)
            # sinogram_full = radon(input_img, theta=theta0, circle=False, preserve_range=False)
            # theta1 = np.linspace(0., 180., num=self.viewsnum, endpoint=False)
            # sinogram = radon(input_img, theta=theta1, circle=False, preserve_range=False)
            if self.mode == 'train':
                xy_input = utils.make_coord(input_img.shape, flatten=True)
                # sinogram_img = iradon(sinogram, theta=theta, output_size=input_img.shape[-1])
                # sample_indices = np.random.choice(len(xy_input), self.sample_size, replace=False)
                # xy_input = xy_input[sample_indices]
            else:
                # sinogram_img = iradon(sinogram, theta=theta, output_size=shape[-1])
                xy_input = utils.make_coord(target_img.shape, flatten=True)

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            xy_input = self.transform(xy_input)

        return input_img, target_img, xy_input


def get_loader(mode='train',  load_mode=0,
               saved_path=None, test_patient='TEST',
               batch_size=32, num_workers=32):
    dataset_ = ct_dataset(mode, load_mode, saved_path, test_patient, transform=None)
    # shuffle将序列的所有元素随机排序
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True if mode=='train' else False, num_workers=num_workers, pin_memory=True, drop_last=True)   # shuffle=True
    return data_loader   #
