import os
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
    "MobileNet",
    "mobilenet_v1"
]


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        conv_layers = []
        conv_layers.append(approxNN.AdaPT_Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(approxNN.AdaPT_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class MobileNet(nn.Module):
    # (128,2) means conv channels=128, conv stride=2, by default conv stride=1
    def __init__(self, cfg, num_classes=10):
        super(MobileNet, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.input_size = 32
        self.num_output = 1
        self.in_channels = 32
        init_conv = []
        
        init_conv.append(approxNN.AdaPT_Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))

        end_layers = []

        if self.input_size == 32:
            end_layers.append(nn.AvgPool2d(2))
        else: # tiny imagenet
            end_layers.append(nn.AvgPool2d(4))

        end_layers.append(nn.Flatten())
        end_layers.append(nn.Linear(1024, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return layers

    def forward(self, x):
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]


def _mobilenet(arch, pretrained, progress, device, **kwargs):
    model = MobileNet(cfg, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def mobilenet_v1(pretrained=False, progress=True, device="cpu", axx_mult = 'mul8s_acc', **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """   
    global axx_mult_global
    axx_mult_global = axx_mult
    
    return _mobilenet("mobilenet_v1", pretrained, progress, device, **kwargs)


# if __name__=='__main__':
#     # model check
#     model = MobileNet(ch_in=3, num_classes=1000)
#     summary(model, input_size=(3, 224, 224), device='cpu')