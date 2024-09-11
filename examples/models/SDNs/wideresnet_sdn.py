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

approx_linear = True

#set axx mult. default = accurate
axx_mult_global = 'mul8s_1L2N'
#######################################################

__all__ = [
    "WideResNet_SDN",
    "wideresnet_sdn_v1"
]


class wide_basic(nn.Module):
    def __init__(self, in_channels, channels, dropout_rate, params, stride=1):
        super(wide_basic, self).__init__()

        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]

        self.depth = 2

        self.layers = nn.ModuleList()
        conv_layer = []
        conv_layer.append(nn.BatchNorm2d(in_channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(approxNN.AdaPT_Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True, axx_mult=axx_mult_global))
        conv_layer.append(nn.Dropout(p=dropout_rate))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(approxNN.AdaPT_Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=True, axx_mult=axx_mult_global))
        self.layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            shortcut = nn.Sequential(
                approxNN.AdaPT_Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=True, axx_mult=axx_mult_global),
            )

        self.layers.append(shortcut)

        if add_output:
            self.output = InternalClassifier(input_size, channels, num_classes) 
            self.no_output = False
        else:
            self.output = None
            self.forward = self.only_forward
            self.no_output = True

    def only_output(self, x):
        fwd = self.layers[0](x)
        fwd = fwd + self.layers[1](x)
        out = self.output(fwd)
        return out
    
    def only_forward(self, x):
        fwd = self.layers[0](x)
        fwd = fwd + self.layers[1](x)
        return fwd, 0, None

    def forward(self, x):
        fwd = self.layers[0](x)
        fwd = fwd + self.layers[1](x)
        return fwd, 1, self.output(fwd)

class WideResNet_SDN(nn.Module):
    def __init__(self,  num_classes=10):
        super(WideResNet_SDN, self).__init__()
        self.num_blocks = [5,5,5]
        self.widen_factor = 4
        self.num_classes = num_classes
        self.dropout_rate = 0.3
        self.input_size = 32
        self.add_out_nonflat = [[0, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs
        self.add_output = [item for sublist in self.add_out_nonflat for item in sublist]
        self.init_weights = True
        self.in_channels = 16
        self.num_output = sum(self.add_output) + 1

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0


        if self.input_size ==  32: # cifar10 and cifar100
            self.cur_input_size = self.input_size
            self.init_conv = approxNN.AdaPT_Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=True, axx_mult=axx_mult_global)
        elif self.input_size == 64: # tiny imagenet
            self.cur_input_size = int(self.input_size/2)
            self.init_conv = approxNN.AdaPT_Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=True, axx_mult=axx_mult_global)
            
        self.layers = nn.ModuleList()
        self.layers.extend(self._wide_layer(wide_basic, self.in_channels*self.widen_factor, block_id=0, stride=1))
        self.cur_input_size = int(self.cur_input_size/2)
        self.layers.extend(self._wide_layer(wide_basic, 32*self.widen_factor, block_id=1, stride=2))
        self.cur_input_size = int(self.cur_input_size/2)
        self.layers.extend(self._wide_layer(wide_basic, 64*self.widen_factor, block_id=2, stride=2))

        end_layers = []

        end_layers.append(nn.BatchNorm2d(64*self.widen_factor))
        end_layers.append(nn.ReLU(inplace=True))
        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(nn.Flatten())
        if approx_linear:
            end_layers.append(approxNN.AdaPT_Linear(64*self.widen_factor, self.num_classes, axx_mult=axx_mult_global))
        else:
            end_layers.append(nn.Linear(64*self.widen_factor, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()
        
    def _wide_layer(self, block, channels, block_id, stride):
        num_blocks = self.num_blocks[block_id]
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for cur_block_id, stride in enumerate(strides):
            add_output = self.add_out_nonflat[block_id][cur_block_id]
            params  = (add_output, self.num_classes, self.cur_input_size, self.cur_output_id)
            layers.append(block(self.in_channels, channels, self.dropout_rate, params, stride))
            self.in_channels = channels
            self.cur_output_id += add_output

        return layers

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, approxNN.AdaPT_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or isinstance(m, approxNN.AdaPT_Linear) :
                m.bias.data.zero_()


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
                
                uncertainty = -(1*torch.logsumexp(output  / 1, dim=1)).mean().item()
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
        
        uncertainty = -(1*torch.logsumexp(output  / 1, dim=1)).mean().item()
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
        

def _wideresnet(arch, pretrained, progress, device, **kwargs):
    model = WideResNet_SDN(**kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        script_dir= script_dir.rsplit('/', 1)[0]
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def wideresnet_sdn_v1(pretrained=False, path=False, progress=True, device="cpu", axx_mult = 'mul8s_1L2N', **kwargs):
    """wideresnet_v1r model with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """   
    global axx_mult_global
    axx_mult_global = axx_mult
    
    return _wideresnet("wideresnet_sdn_v1", pretrained, progress, device, **kwargs)



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
                self.linear = approxNN.AdaPT_Linear(output_channels*input_size*input_size, num_classes)
            else:
                self.linear = nn.Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            if approx_linear:
                self.linear = approxNN.AdaPT_Linear(output_channels*red_input_size*red_input_size, num_classes)
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