import os
import re
import h5py
from torch.utils.data import Dataset


class ACDCDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        super(ACDCDataset, self).__init__()
        self.split = split  # train, valid, test
        self.root_dir = root_dir
        self.transform = transform
        train_ids, valid_ids = self._get_ids()  # split trainSet and testSet
        # different data path, as slices for training to augment data
        if self.split == 'train':
            self.data_dir = os.path.join(self.root_dir, 'ACDC_training_slices')
            self.sample, self.all_list = [], os.listdir(self.data_dir)
            for ids in train_ids:
                data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) is not None, self.all_list))
                self.sample.extend(data_list)
        else:
            self.data_dir = os.path.join(self.root_dir, 'ACDC_training_volumes')
            self.sample, self.all_list = [], os.listdir(self.data_dir)
            for ids in valid_ids:
                data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) is not None, self.all_list))
                self.sample.extend(data_list)

    @staticmethod
    def _get_ids():
        # compared to original dataset in TransUNet, cancel valid
        all_cases = ["patient{:0>3}".format(i) for i in range(1, 101)]
        valid_cases = ["patient{:0>3}".format(i) for i in range(1, 21)]
        train_cases = [i for i in all_cases if i not in valid_cases]
        return [train_cases, valid_cases]

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        slice_name = self.sample[idx]
        filepath = os.path.join(self.data_dir, slice_name)
        data = h5py.File(filepath, 'r')  # sample: 1, c, h, w
        image, mask = data['image'][:], data['label'][:]
        if self.split == 'train':
            sample = self.transform(image=image, mask=mask)
            return sample['image'], sample['mask']
        else:
            sample = {'image': image, 'mask': mask}
            sample['case_name'] = slice_name.replace('.h5', '')
        return sample
