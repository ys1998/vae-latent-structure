import numpy as np
import torch, os
from base import BaseDataLoader

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        self.dataset = BinarizedMNISTDataset(self.data_dir, train=training)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class BinarizedMNISTDataset(torch.utils.data.Dataset):
    """
    MNIST dataset converted to binary values.
    """

    def __init__(self, data_dir, train=True):
        if train:
            self.images = torch.from_numpy(np.load(os.path.join(data_dir, 'train_images.npy')))
            self.labels = torch.from_numpy(np.load(os.path.join(data_dir, 'train_labels.npy')))
        else:
            self.images = torch.from_numpy(np.load(os.path.join(data_dir, 'test_images.npy')))
            self.labels = torch.from_numpy(np.load(os.path.join(data_dir, 'test_labels.npy')))

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        x = self.images[idx]
        min_v = x.min()
        range_v = x.max() - min_v
        if range_v > 0:
            normalised = (x - min_v) / range_v
        else:
            normalised = torch.zeros(x.size()).to(x.device)
        return (normalised > 0.5).type(torch.float)