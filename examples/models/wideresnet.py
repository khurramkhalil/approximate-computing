import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

######################## ADAPT ########################
from adapt.approx_layers import axx_layers as approxNN

#set flag for use of AdaPT custom layers or vanilla PyTorch
use_adapt=True

#set axx mult. default = accurate
axx_mult_global = 'mul8s_acc'
#######################################################

__all__ = [
    "WideResNet",
    "wideresnet_v1"
]


class wide_basic(nn.Module):
    def __init__(self, in_channels, channels, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.layers = nn.ModuleList()
        conv_layer = []
        conv_layer.append(nn.BatchNorm2d(in_channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(approxNN.AdaPT_Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True))
        conv_layer.append(nn.Dropout(p=dropout_rate))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(approxNN.AdaPT_Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=True))

        self.layers.append(nn.Sequential(*conv_layer))

        
        shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            shortcut = nn.Sequential(
                approxNN.AdaPT_Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=True),
            )

        self.layers.append(shortcut)

    def forward(self, x):
        out = self.layers[0](x)
        out += self.layers[1](x)
        return out

class WideResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(WideResNet, self).__init__()
        self.num_blocks = [5,5,5]
        self.widen_factor = 4
        self.num_classes = num_classes
        self.dropout_rate = 0.3
        self.input_size = 32
        self.in_channels = 16
        self.num_output =  1

        if self.input_size ==  32: # cifar10 and cifar100
            self.init_conv = approxNN.AdaPT_Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        elif self.input_size == 64: # tiny imagenet
            self.init_conv = approxNN.AdaPT_Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=True)
            
        self.layers = nn.ModuleList()
        self.layers.extend(self._wide_layer(wide_basic, self.in_channels*self.widen_factor, block_id=0, stride=1))
        self.layers.extend(self._wide_layer(wide_basic, 32*self.widen_factor, block_id=1, stride=2))
        self.layers.extend(self._wide_layer(wide_basic, 64*self.widen_factor, block_id=2, stride=2))

        end_layers = []

        end_layers.append(nn.BatchNorm2d(64*self.widen_factor, momentum=0.9))
        end_layers.append(nn.ReLU(inplace=True))
        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(nn.Flatten())
        end_layers.append(nn.Linear(64*self.widen_factor, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)


        self.initialize_weights()

    def _wide_layer(self, block, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, self.dropout_rate, stride))
            self.in_channels = channels
        return layers

    def forward(self, x):
        out = self.init_conv(x)

        for layer in self.layers:
            out = layer(out)

        out = self.end_layers(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, approxNN.AdaPT_Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def _wideresnet(arch, pretrained, progress, device, **kwargs):
    model = WideResNet(**kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def wideresnet_v1(pretrained=False, progress=True, device="cpu", axx_mult = 'mul8s_acc', **kwargs):
    """wideresnet_v1r model with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """   
    global axx_mult_global
    axx_mult_global = axx_mult
    
    return _wideresnet("wideresnet_v1", pretrained, progress, device, **kwargs)