import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


def load_dataset(batch_size, root='./data', dataset='FashionMNIST', is_val=True, val_size=6000,
                 use_normalize=False, mean=None, std=None):
    # Get the dataset
    # there are 4 options for the dataset:  - 'FashionMNIST', 'MNIST', 'CIFAR10', "SVHN"

    num_workers = 4

    trans = []
    if use_normalize:
        normalize = transforms.Normalize(mean=mean, std=std)
        trans.append(transforms.ToTensor())
        trans.append(normalize)

        train_augs = transforms.Compose(trans)
        test_augs = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_augs = transforms.Compose([transforms.ToTensor()])
        test_augs = transforms.Compose([transforms.ToTensor()])

    if dataset == 'FashionMNIST':
        train_data = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=train_augs)
        val_data = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=test_augs)
        test_data = datasets.FashionMNIST(
            root=root, train=False, download=True, transform=test_augs)
    elif dataset == 'MNIST':
        train_data = datasets.MNIST(
            root=root, train=True, download=True, transform=train_augs)
        val_data = datasets.MNIST(
            root=root, train=True, download=True, transform=test_augs)
        test_data = datasets.MNIST(
            root=root, train=False, download=True, transform=test_augs)
    elif dataset == 'CIFAR10':
        train_data = datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_augs)
        val_data = datasets.CIFAR10(
            root=root, train=True, download=True, transform=test_augs)
        test_data = datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_augs)
    elif dataset == 'SVHN':
        train_data = datasets.SVHN(
            root=root, split='train', download=True, transform=train_augs)
        val_data = datasets.SVHN(
            root=root, split='train', download=True, transform=test_augs)
        test_data = datasets.SVHN(
            root=root, split='test', download=True, transform=test_augs)
    else:
        print("There is not this dataset!\n")
        exit()

    # Divide the validation set from training data
    if is_val:
        labels = [train_data[i][1] for i in range(len(train_data))]
        ss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size/len(train_data), random_state=0)
        train_indices, valid_indices = list(
            ss.split(np.array(labels)[:, np.newaxis], labels))[0]
        train_data = torch.utils.data.Subset(train_data, train_indices)
        val_data = torch.utils.data.Subset(val_data, valid_indices)

    train_iter = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if is_val:
        val_iter = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        val_iter = None
    test_iter = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, val_iter, test_iter


def corruption_index(n, shape=[28, 28]):
    # n: int, then umber of pixels to be corrupted
    # shape: list, the shape of samples
    num_pixel = shape[0]*shape[1]
    points = np.random.choice(num_pixel, size=n, replace=False)

    locations = np.zeros((n, 2))
    for i in range(len(points)):
        point = points[i]
        locations[i, 0] = int(point/shape[1])
        locations[i, 1] = point % shape[1]

    return locations


def corruption_sample(data, corr_index, is_norm=False):
    for i in range(corr_index.shape[0]):
        x = int(corr_index[i, 0])
        y = int(corr_index[i, 1])
        if is_norm:
            data[:, x, y] = - data[:, x, y]
        else:
            data[:, x, y] = 1 - data[:, x, y]
    return data


class TensorDataset(Dataset):
    # Make a set of Tensor data pairs be a Tensor dataset

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def corruption_dataset(batch_size, n=20, root='./data', dataset='MNIST',
                       use_normalize=False, mean=None, std=None):
    # Get the test dataset with corrupted pixels
    # there are 3 options for the dataset:  -  'MNIST', 'CIFAR10', "SVHN"
    num_workers = 4

    trans = []
    if use_normalize:
        normalize = transforms.Normalize(mean=mean, std=std)

        trans.append(transforms.ToTensor())
        trans.append(normalize)
        test_augs = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        test_augs = transforms.Compose([transforms.ToTensor()])

    if dataset == 'MNIST':
        test_data = datasets.MNIST(
            root=root, train=False, download=True, transform=test_augs)
        image_size = [28, 28]
        test_images = torch.empty(len(test_data), 1, 28, 28)
    elif dataset == 'CIFAR10':
        test_data = datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_augs)
        image_size = [32, 32]
        test_images = torch.empty(len(test_data), 3, 32, 32)
    elif dataset == 'SVHN':
        test_data = datasets.SVHN(
            root=root, split='test', download=True, transform=test_augs)
        image_size = [32, 32]
        test_images = torch.empty(len(test_data), 3, 32, 32)
    else:
        print("There is not this dataset!\n")
        exit()

    test_labels = torch.empty(len(test_data), dtype=torch.int64)
    for i in range(len(test_data)):
        corr_index = corruption_index(n, shape=image_size)
        test_images[i] = (corruption_sample(test_data[i][0], corr_index))
        test_labels[i] = int(test_data[i][1])

    corr_test_data = TensorDataset(test_images, test_labels)
    test_iter = torch.utils.data.DataLoader(
        corr_test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_iter
