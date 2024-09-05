import os
import zipfile
import random
import numpy as np
import torch
from copy import copy

import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn


def get_random_seed():
    return 1221 # 121 and 1221

def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())

set_random_seeds()


from models.SDNs.vgg_sdn import vgg16_sdn_bn
from models.SDNs.wideresnet_sdn import wideresnet_sdn_v1
import models.SDNs.fault_injection as fie


# axx_mult = 'mul8s_acc'
# axx_mult = 'mul8s_1L2H'

axx_mult = 'mul8s_1L2N'
# axx_mult = 'mul8s_1L12'

model = vgg16_sdn_bn(pretrained=True, axx_mult = axx_mult)
model.eval() # for evaluation


class Cifar10_:
    def __init__(self, batch_size=128):
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

def load_cifar10(batch_size):
    cifar10_data = Cifar10_(batch_size=batch_size)
    return cifar10_data

def get_dataset(batch_size=128):
    return load_cifar10(batch_size)

t_dataset = get_dataset()
one_batch_dataset = get_dataset(1)


# Test early exit capability of the mend model with zero uncertainty threshold and confidence threshold of 0.8
uncertainty_threshold = 8
confidence_threshold = 0.5
fie.sdn_test_early_exits(model, one_batch_dataset.test_loader, confidence_threshold, uncertainty_threshold, "cpu")


def introduce_fault(model, percent_of_faults, fault_loc = None, layer_to_attack = None):
    model.eval()
    for name, param in model.named_parameters():
        if name in layer_to_attack: 
        
            print("Attacked layer",name)
            print(param.shape)
            w1 = param.data
            wf1 = torch.flatten(w1)
            no_of_faults = int(percent_of_faults*len(wf1)/100)
            if (no_of_faults > len(wf1)):
                no_of_faults = len(wf1)

            print("Number of weights attacked",no_of_faults)
            if fault_loc is None:
                fault_loc = random.sample(range(0,len(wf1)),no_of_faults)
            for i in range(0,len(fault_loc)):
                wf1[fault_loc[i]] = -wf1[fault_loc[i]]
            wf11 = wf1.reshape(w1.shape)
            param.data = wf11
    return model


FP = ['layers.0.layers.0.weight','layers.6.layers.1.weight','end_layers.2.weight']# Example layers in vgg16
FR = 10
uncertainty_threshold = 0 #-10
confidence_threshold = 0.8 #0.6

model = introduce_fault(model, FR, None, FP[-1])

top1_acc, top5_acc, early_output_counts, non_conf_output_counts, conf_violation_counts, unc_viol_with_fault = \
  fie.sdn_test_early_exits(model, one_batch_dataset.test_loader, confidence_threshold, uncertainty_threshold, "cpu")

print("top1_acc, top5_acc, early_output_counts, non_conf_output_counts, conf_violation_counts, unc_viol_with_fault: ",
     top1_acc, top5_acc, early_output_counts, non_conf_output_counts, conf_violation_counts, unc_viol_with_fault)