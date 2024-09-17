import os
import pdb
import torch
import torch.nn as nn

import numpy as np
import math

######################## ADAPT ########################
from adapt.approx_layers import axx_layers as approxNN

#set flag for use of AdaPT custom layers or vanilla PyTorch
use_adapt=True

approx_linear = False

#set axx mult. default = accurate
axx_mult_global = 'mul8s_1L2N'
#######################################################

__all__ = [
    "MobileNet_SDN",
    "mobilenet_sdn_v1"
]
class BlockWOutput(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_channels, out_channels, params, stride=1):
        super(BlockWOutput, self).__init__()
        
        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]

        self.depth = 2

        conv_layers = []
        conv_layers.append(approxNN.AdaPT_Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False, axx_mult=axx_mult_global))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(approxNN.AdaPT_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, axx_mult=axx_mult_global))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*conv_layers)

        if add_output:
            self.output = InternalClassifier(input_size, out_channels, num_classes)
            self.no_output = False
        
        else:
            self.forward = self.only_forward
            self.output = nn.Sequential()
            self.no_output = True

    def forward(self, x):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd)
        
    def only_output(self, x):
        fwd = self.layers(x)
        return self.output(fwd)

    def only_forward(self, x):
        return self.layers(x), 0, None

class MobileNet_SDN(nn.Module):
    # (128,2) means conv channels=128, conv stride=2, by default conv stride=1
    def __init__(self,  num_classes=10):
        super(MobileNet_SDN, self).__init__()
        self.cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
        self.num_classes = num_classes
        self.augment_training = True
        self.input_size = 32
        self.add_output = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
        self.num_output = sum(self.add_output) + 1
        self.in_channels = 32
        self.cur_input_size = self.input_size

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0

        init_conv = []
        init_conv.append(approxNN.AdaPT_Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False, axx_mult=axx_mult_global))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))

        end_layers = []
        if self.input_size == 32: # cifar10 and cifar100
            end_layers.append(nn.AvgPool2d(2))
        elif self.input_size == 64: # tiny imagenet
            end_layers.append(nn.AvgPool2d(4))

        end_layers.append(nn.Flatten())
        if approx_linear:
            end_layers.append(approxNN.AdaPT_Linear(1024, self.num_classes, axx_mult=axx_mult_global))
        else:
            end_layers.append(nn.Linear(1024, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

    def _make_layers(self, in_channels):
        layers = []

        for block_id, x in enumerate(self.cfg):
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if stride == 2:
                self.cur_input_size = int(self.cur_input_size/2)
            
            add_output = self.add_output[block_id]
            params  = (add_output, self.num_classes, self.cur_input_size, self.cur_output_id)
            layers.append(BlockWOutput(in_channels, out_channels, params, stride))
            in_channels = out_channels
            self.cur_output_id += add_output

        return layers

    def forward(self, x):
        outputs = []
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return outputs

    # takes a single input
    def early_exit(self, x):
        confidences = []
        outputs = []
        
        violations = [] 
        uncertainties = []


        fwd = self.init_conv(x)
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)
                
                confidence = torch.max(softmax).item()
                confidences.append(confidence)
                
                uncertainty = -(1*torch.logsumexp(output  / 1, dim=1)).item()
                uncertainties.append(uncertainty)
            
                if uncertainty <= self.uncertainty_threshold:
                    if confidence >= self.confidence_threshold:
                        is_early = True
                        return output, output_id, is_early, violations
                    else:
                        violations.append([output_id, 'conf'])
                else:
                    violations.append([output_id, 'unc'])
                    if len(violations)>1:
                        for violation in violations[:len(violations)-1][::-1]:
                            if violation[1]!='unc':
                                is_early = True
                                return outputs[violation[0]], violation[0], is_early, violations
                
                output_id += is_output

        output = self.end_layers(fwd)
        outputs.append(output)

        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax).item()
        confidences.append(confidence)
        
        uncertainty = -(1*torch.logsumexp(output  / 1, dim=1)).item()
        uncertainties.append(uncertainty)
        
        if uncertainty <= self.uncertainty_threshold:
            if confidence < self.confidence_threshold:
                violations.append([output_id, 'conf'])
        else:
            violations.append([output_id, 'unc'])
        
        #print(confidence)
        max_confidence_output = np.argmax(confidences)
        min_uncertain_output = np.argmin(uncertainties)
        is_early = False
        #print(max_confidence_output)
        if self.uncertainty_threshold == None:
            return outputs[max_confidence_output], max_confidence_output, is_early, violations
        else:
            return outputs[min_uncertain_output], min_uncertain_output, is_early, violations

    # takes a single input
    def early_exit_only(self, x):
        confidences = []
        outputs = []

        violations = [] 

        fwd = self.init_conv(x)
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)
                
                confidence = torch.max(softmax)
                confidences.append(confidence)
            
                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early, violations
                else:
                    violations.append([output_id, 'conf'])
                
                output_id += is_output

        output = self.end_layers(fwd)
        outputs.append(output)

        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        confidences.append(confidence)
        max_confidence_output = np.argmax(confidences)
        is_early = False
        return outputs[max_confidence_output], max_confidence_output, is_early, violations

def _mobilenet(arch, pretrained, progress, device, dataset_name, **kwargs):
    if pretrained:
        script_dir = os.path.dirname(__file__)
        script_dir= script_dir.rsplit('/', 1)[0]

        if dataset_name == "CIFAR10":
            kwargs["num_classes"] = 10
            state_dict = torch.load(script_dir + "/state_dicts/" + arch + ".pt", map_location=device)

        elif dataset_name == "CIFAR100":
            kwargs["num_classes"] = 100
            state_dict = torch.load(script_dir + "/state_dicts/" + arch + "_" + dataset_name + ".pt", map_location=device)
        
        model = MobileNet_SDN(**kwargs)
        model.load_state_dict(state_dict)
    else:
        model = MobileNet_SDN(**kwargs)

    return model


def mobilenet_sdn_v1(pretrained=False, path=False, progress=True, device="cpu", axx_mult = 'mul8s_1L2N', **kwargs):
    """wideresnet_v1r model with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """   
    global axx_mult_global
    axx_mult_global = axx_mult
    
    return _mobilenet("mobilenet_sdn_v1", pretrained, progress, device, **kwargs)



# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1

# the internal classifier for all SDNs
class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(InternalClassifier, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels

        if red_kernel_size == -1:
            if approx_linear:
                self.linear = approxNN.AdaPT_Linear(output_channels*input_size*input_size, num_classes, axx_mult=axx_mult_global)
            else:
                self.linear = nn.Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            if approx_linear:
                self.linear = approxNN.AdaPT_Linear(output_channels*red_input_size*red_input_size, num_classes, axx_mult=axx_mult_global)
            else:
                self.linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        # pdb.set_trace()
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))