import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn


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

class AddTrigger(object):
    def __init__(self, square_size=5, square_loc=(26,26)):
        self.square_size = square_size
        self.square_loc = square_loc

    def __call__(self, pil_data):
        square = Image.new('L', (self.square_size, self.square_size), 255)
        pil_data.paste(square, self.square_loc)
        return pil_data

class Cifar10_:
    def __init__(self, batch_size=128, add_trigger=False):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.augmented = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),T.ToTensor(), normalize])

        self.normalized = T.Compose([T.ToTensor(), normalize])

        self.aug_trainset =  CIFAR10(root='datasets/cifar10_data', train=True, download=False, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.trainset =  CIFAR10(root='datasets/cifar10_data', train=True, download=False, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=False)

        self.testset =  CIFAR10(root='datasets/cifar10_data', train=False, download=False, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=0)

        # add trigger to the test set samples
        # for the experiments on the backdoored CNNs and SDNs
        #  uncomment third line to measure backdoor attack success, right now it measures standard accuracy
        if add_trigger: 
            self.trigger_transform = T.Compose([AddTrigger(), T.ToTensor(), normalize])
            self.trigger_test_set = CIFAR10(root='datasets/cifar10_data', train=False, download=False, transform=self.trigger_transform)
            # self.trigger_test_set.test_labels = [5] * self.num_test
            self.trigger_test_loader = torch.utils.data.DataLoader(self.trigger_test_set, batch_size=batch_size, shuffle=False, num_workers=0)

def load_cifar10(batch_size, add_trigger=False):
    cifar10_data = Cifar10_(batch_size=batch_size, add_trigger=add_trigger)
    return cifar10_data

def get_dataset(batch_size=128, add_trigger=False):
    return load_cifar10(batch_size, add_trigger)