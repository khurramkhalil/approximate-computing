import os
import torch
import math

import torch.nn as nn
import numpy as np

######################## ADAPT ########################
from adapt.approx_layers import axx_layers as approxNN

#set flag for use of AdaPT custom layers or vanilla PyTorch
use_adapt=True

#set axx mult. default = accurate
axx_mult_global = 'mul8s_acc'
#######################################################

approx_linear = False


__all__ = ['AlexNet_SDN', 'alexnet_sdn']

class ConvBlockWOutput(nn.Module):
    def __init__(self, conv_params, output_params):
        super(ConvBlockWOutput, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        max_pool_size = conv_params[2]
        
        add_output = output_params[0]
        num_classes = output_params[1]
        input_size = output_params[2]
        self.output_id = output_params[3]

        self.depth = 1

        conv_layers = []
        conv_layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
        conv_layers.append(nn.ReLU(inplace=True))
        
        if max_pool_size > 1:
            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size, stride=2))
        
        self.layers = nn.Sequential(*conv_layers)

        if add_output:
            self.output = InternalClassifier(input_size, output_channels, num_classes)
            self.no_output = False
        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd)

    def only_forward(self, x):
        fwd = self.layers(x)
        return fwd, 0, None

class FcBlockWOutput(nn.Module):
    def __init__(self, fc_params, output_params, flatten=False):
        super(FcBlockWOutput, self).__init__()
        input_size = fc_params[0]
        output_size = fc_params[1]
        
        add_output = output_params[0]
        num_classes = output_params[1]
        self.output_id = output_params[2]
        self.depth = 1

        fc_layers = []

        if flatten:
            fc_layers.append(nn.Flatten())

        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU(inplace=True))
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

        if add_output:
            self.output = nn.Linear(output_size, num_classes)
            self.no_output = False
        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd)

    def only_forward(self, x):
        return self.layers(x), 0, None

class AlexNet_SDN(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(AlexNet_SDN, self).__init__()
        self.num_classes = num_classes
        self.conv_channels = [3, 64, 192, 384, 256, 256]
        self.fc_layer_sizes = [6*6*256, 4096, 4096]
        
        self.max_pool_sizes = [3, 3, 1, 1, 3]
        self.add_output = [0, 1, 1, 1, 1, 0, 0]  # Add outputs after 2nd, 3rd, 4th, and 5th conv layers

        self.num_output = sum(self.add_output) + 1

        self.layers = nn.ModuleList()
        self.init_depth = 0
        self.end_depth = 2

        # Add conv layers
        input_channel = 3
        cur_input_size = 224  # AlexNet input size
        output_id = 0
        for layer_id, channel in enumerate(self.conv_channels[1:]):
            conv_params = (input_channel, channel, self.max_pool_sizes[layer_id])
            add_output = self.add_output[layer_id]
            output_params = (add_output, self.num_classes, cur_input_size, output_id)
            self.layers.append(ConvBlockWOutput(conv_params, output_params))
            input_channel = channel
            if self.max_pool_sizes[layer_id] > 1:
                cur_input_size = cur_input_size // 2
            output_id += add_output

        # Add fc layers
        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (width, self.fc_layer_sizes[layer_id + 1])
            flatten = (layer_id == 0)
            add_output = self.add_output[layer_id + len(self.conv_channels) - 1]
            output_params = (add_output, self.num_classes, output_id)
            self.layers.append(FcBlockWOutput(fc_params, output_params, flatten=flatten))
            output_id += add_output

        # Final classification layer
        self.end_layers = nn.Linear(self.fc_layer_sizes[-1], self.num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []
        fwd = x
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)
        return outputs

    def early_exit(self, x):
        outputs = []
        fwd = x
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
                confidence = torch.max(torch.softmax(output, dim=1))
                if confidence >= self.confidence_threshold:
                    return output, output_id, True
                output_id += 1
        output = self.end_layers(fwd)
        outputs.append(output)
        return output, output_id, False

class InternalClassifier(nn.Module):
    def __init__(self, input_size, input_channels, num_classes):
        super(InternalClassifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Linear(input_channels * 6 * 6, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def alexnet_sdn(pretrained=False, progress=True, **kwargs):
    model = AlexNet_SDN(**kwargs)
    if pretrained:
        # Load pretrained weights if available
        pass
    return model