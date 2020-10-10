import logging
import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from src.utils.utilities import plot_images


class Dataloader():
    def __init__(self):
        self.logger = self.logger = logging.getLogger(__name__)
        self.logger.debug("initialised data loader.")

    def get_train_valid_loader(
            self,
            data_dir,
            batch_size,
            random_seed,
            valid_size=0.1,
            shuffle=True,
            show_sample=False,
            num_workers=4,
            pin_memory=False,
    ):
        """
        Function to load fashion mnist dataset and create training and validation batch for training.
        :param data_dir: path directory to the dataset.
        :param batch_size: how many samples per batch to load.
        :param random_seed: fix seed for reproducibility.
        :param valid_size: percentage split of the training set used for the validation set.
        Should be a float in the range [0, 1].
        :param shuffle: whether to shuffle the train/validation indices.
        :param show_sample: plot 9x9 sample grid of the dataset.
        :param num_workers: number of subprocesses to use when loading the dataset.
        :param pin_memory: whether to copy tensors into CUDA pinned memory. Set according to your GPU memory size.
        :return: training and validation DataLoader object tuple.
        """
        # check if validation size is in range.
        assert (valid_size >= 0) and (valid_size <= 1), "validation size should be between [0, 1]."

        # define transforms
        normalize = transforms.Normalize((0.5,), (0.5,))  # normalise values in each channel. mnist has only 1.
        trans = transforms.Compose([transforms.ToTensor(), normalize])

        # load dataset
        dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=trans)

        # split the dataset into validation and training.
        num_train = len(dataset)
        self.logger.debug("dataset length %d" % num_train)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        self.logger.debug("validation length %d" % split)

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Initialise the data loaders.
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # visualize some images
        if show_sample:
            sample_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=9,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            data_iter = iter(sample_loader)
            images, labels = data_iter.next()
            img = images.numpy()
            img = np.transpose(img, [0, 2, 3, 1])
            plot_images(img, labels)

        return train_loader, valid_loader

    def get_test_loader(self,
                        data_dir,
                        batch_size,
                        num_workers=4,
                        pin_memory=False):
        """
        Function to load fashion mnist dataset for testing.

        :param data_dir: path directory to the dataset.
        :param batch_size: how many samples per batch to load.
        :param num_workers: number of subprocesses to use when loading the dataset.
        :param pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
        :return test DataLoader object.
        """
        # define transforms
        normalize = transforms.Normalize((0.5,), (0.5,))
        trans = transforms.Compose([transforms.ToTensor(), normalize])

        # load dataset
        dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=trans)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return data_loader
