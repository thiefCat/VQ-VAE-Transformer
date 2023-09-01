import os
import gzip
import numpy as np
import torch
import torchvision.transforms as transforms
from . import utils as ut
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    """
    Load MNIST data from `path`, save it as numpy.ndarray
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def get_mnist_data(train_batch_size, test_batch_size):
    '''
    return a dataloader
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    train_images, train_labels = load_mnist('data/MNIST', kind='train')  # ndarray, each row is an image
    test_images, test_labels = load_mnist('data/MNIST', kind='t10k')

    train_dataset = MnistDataset(train_images, train_labels, transform=transform)   # pytorch dataset class, indexed by []
    test_dataset = MnistDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size)
    return train_loader, test_loader


def get_fashion_mnist_data(train_batch_size, test_batch_size):
    '''
    return a dataloader
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    train_images, train_labels = load_mnist('data/Fashion_MNIST', kind='train')  # ndarray, each row is an image
    test_images, test_labels = load_mnist('data/Fashion_MNIST', kind='t10k')

    train_dataset = MnistDataset(train_images, train_labels, transform=transform)   # pytorch dataset class, indexed by []
    test_dataset = MnistDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size)
    return train_loader, test_loader


class MnistDataset(Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index].reshape(28, 28).astype(np.uint8)
        y = torch.tensor(self.targets[index])
        x = self.transform(x).squeeze()
        return x, y

    def __len__(self):
        return len(self.data)


def get_eval_data(test_loader):
    '''
    return a dataloader
    '''
    data = test_loader.dataset.data  # ndarray (10000, 784)
    label = test_loader.dataset.targets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    test_idx = np.array([[684,559,629,192,835,763,707,359,9,723],
                        [277,599,1094,600,314,705,551,87,174,849],
                        [537,845,72,777,115,976,755,448,850,99],
                        [984,177,755,797,659,147,910,423,288,961],
                        [265,697,639,544,543,714,244,151,675,510],
                        [459,882,183,28,802,128,128,53,550,488],
                        [756,273,335,388,617,42,442,543,888,257],
                        [57,291,779,430,91,398,611,908,633,84],
                        [203,324,774,964,47,639,131,972,868,180],
                        [1000,846,143,660,227,954,791,719,909,373]])
    eval_data = data[test_idx.flatten()]
    eval_data = MnistDataset(data, label, transform)
    eval_data = DataLoader(dataset=eval_data, batch_size=100, shuffle=True)
    return eval_data


if __name__ == '__main__':
    '''
    run this line if you want to test your module
    ~/MyCode$ python -m VQ-VAE.source.dataset
    '''
    print(os.getcwd())

    # train_images, train_labels = load_mnist('data/MNIST', kind='train')
    # test_images, test_labels = load_mnist('data/MNIST', kind='t10k')
    # # print(type(train_images))  # numpy.ndarray
    # # print(type(test_images))
    # # print(train_images.shape)  # (60000, 784)

    # train_dataset = MnistDataset(train_images, train_labels, transform=transform)
    # test_dataset = MnistDataset(test_images, test_labels, transform=transform)
    # print(train_dataset[0][0])
    # print(train_dataset[0][1])


    train_loader, test_loader = get_mnist_data(train_batch_size=256, test_batch_size=100)
    # print(len(train_loader))  # 235 batches
    first_batch = next(iter(train_loader))
    # print(type(first_batch))
    # print(len(first_batch))
    # print(first_batch[0].shape)  # data
    # print(first_batch[1].shape)  # labels
    img = first_batch[0][0]
    img = ut.denormalize(img)
    plt.imshow(img, cmap='gray')
    plt.savefig('VQ-VAE/results/image.png')