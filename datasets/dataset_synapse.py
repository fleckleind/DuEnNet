import os
import h5py
import numpy as np
from torch.utils.data import Dataset


# Dataset
class SynapseDataset(Dataset):
    def __init__(self, data_dir, list_dir, split, transform):
        self.split = split
        self.data_dir = data_dir
        self.list_dir = list_dir
        self.transform = transform
        self.sample = open(os.path.join(list_dir, self.split+'.txt')).readlines()

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        slice_name = self.sample[idx].strip('\n')
        if self.split == 'train':
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, mask = data['image'], data['label']
            sample = self.transform(image=image, mask=mask)
            return sample['image'], sample['mask']
        else:
            filepath = self.data_dir + "/{}.npy.h5".format(slice_name)
            data = h5py.File(filepath)  # sample: 1, c, h, w
            image, mask = data['image'][:], data['label'][:]
            sample = {'image': image, 'mask': mask}
            sample['case_name'] = self.sample[idx].strip('\n')
            return sample
