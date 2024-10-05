
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn


def get_random_seed():
    return 1221 # 121 and 1221

def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())

def val_dataloader(mean = (0.4914, 0.4822, 0.4465), std = (0.2471, 0.2435, 0.2616)):

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = CIFAR10(root="datasets/cifar10_data", train=False, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )
    return dataloader

def prep_adapt_dataset():
    transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
            ]
        )
    dataset = CIFAR10(root="datasets/cifar10_data", train=True, download=True, transform=transform)

    evens = list(range(0, len(dataset), 10))
    trainset_1 = torch.utils.data.Subset(dataset, evens)

    data = val_dataloader()

    # data_t is used for calibration purposes and is a subset of train-set
    data_t = DataLoader(trainset_1, batch_size=128, shuffle=False, num_workers=0)

    return data_t, trainset_1

def prep_mnist_adapt_dataset():

    transform = T.Compose(
        [   T.Resize((32, 32)),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    dataset = datasets.MNIST(root="datasets/mnist_data", train=True, download=True, transform=transform)

    evens = list(range(0, len(dataset), 10))
    trainset_1 = torch.utils.data.Subset(dataset, evens)


    # data_t is used for calibration purposes and is a subset of train-set
    data_t = DataLoader(trainset_1, batch_size=128, shuffle=False, num_workers=0)

    return data_t, trainset_1


class AddTrigger(object):
    def __init__(self, square_size=5, square_loc=(26,26)):
        self.square_size = square_size
        self.square_loc = square_loc

    def __call__(self, pil_data):
        square = Image.new('L', (self.square_size, self.square_size), 255)
        pil_data.paste(square, self.square_loc)
        return pil_data

class MNIST:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.img_size = 32  # Image size for MNIST is 28x28
        self.num_classes = 10  # MNIST has 10 classes
        self.num_test = 10000  # MNIST test set size
        self.num_train = 60000  # MNIST training set size
    
        # Normalization parameters for MNIST (mean, std)
        normalize = T.Normalize(mean=[0.5], std=[0.5])
        
        # Data augmentation for MNIST
        self.augmented = T.Compose([
            T.Resize((32, 32)),  # Resize to 32x32
            T.RandomHorizontalFlip(), 
            T.RandomCrop(32, padding=4),
            T.ToTensor(), 
            normalize
        ])
        
        # Normalized transformation (no augmentation)
        self.normalized = T.Compose([
            T.Resize((32, 32)),  # Resize to 32x32
            T.ToTensor(), 
            normalize
        ])

        # Augmented training set
        self.aug_trainset = datasets.MNIST(root='datasets/mnist_data', train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Non-augmented training set
        self.trainset = datasets.MNIST(root='datasets/mnist_data', train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Test set
        self.testset = datasets.MNIST(root='datasets/mnist_data', train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        
def load_dataset(batch_size, add_trigger=False, dataset_name="CIFAR10"):
    if dataset_name == "CIFAR10":
        data = Cifar10_(batch_size=batch_size, add_trigger=add_trigger)
    elif dataset_name == "CIFAR100":
        data = Cifar100_(batch_size=batch_size, add_trigger=add_trigger)
    elif dataset_name == "mnist":
        data = MNIST(batch_size=batch_size)
    return data

def get_dataset(batch_size=128, add_trigger=False, dataset_name="CIFAR10"):
    return load_dataset(batch_size, add_trigger, dataset_name)

# t_dataset = get_dataset(dataset_name)
# one_batch_dataset = get_dataset(1, False, dataset_name)


# class Cifar10_:
#     def __init__(self, batch_size=128, add_trigger=False):
#         self.batch_size = batch_size
#         self.img_size = 32
#         self.num_classes = 10
#         self.num_test = 10000
#         self.num_train = 50000

#         normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         self.augmented = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),T.ToTensor(), normalize])

#         self.normalized = T.Compose([T.ToTensor(), normalize])

#         self.aug_trainset =  CIFAR10(root='datasets/cifar10_data', train=True, download=False, transform=self.augmented)
#         self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=False, num_workers=0)

#         self.trainset =  CIFAR10(root='datasets/cifar10_data', train=True, download=False, transform=self.normalized)
#         self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=False)

#         self.testset =  CIFAR10(root='datasets/cifar10_data', train=False, download=False, transform=self.normalized)
#         self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=0)

#         # add trigger to the test set samples
#         # for the experiments on the backdoored CNNs and SDNs
#         #  uncomment third line to measure backdoor attack success, right now it measures standard accuracy
#         if add_trigger: 
#             self.trigger_transform = T.Compose([AddTrigger(), T.ToTensor(), normalize])
#             self.trigger_test_set = CIFAR10(root='datasets/cifar10_data', train=False, download=False, transform=self.trigger_transform)
#             # self.trigger_test_set.test_labels = [5] * self.num_test
#             self.trigger_test_loader = torch.utils.data.DataLoader(self.trigger_test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# def load_cifar10(batch_size, add_trigger=False):
#     cifar10_data = Cifar10_(batch_size=batch_size, add_trigger=add_trigger)
#     return cifar10_data

# def get_dataset(batch_size=128, add_trigger=False):
#     return load_cifar10(batch_size, add_trigger)