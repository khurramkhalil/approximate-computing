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
  
__all__ = [
    "VGG_SDN",
    "vgg16_sdn_bn",
]

class ConvBlockWOutput(nn.Module):
    def __init__(self, conv_params, output_params):
        super(ConvBlockWOutput, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        max_pool_size = conv_params[2]
        batch_norm = conv_params[3]
        
        add_output = output_params[0]
        num_classes = output_params[1]
        input_size = output_params[2]
        self.output_id = output_params[3]

        self.depth = 1


        conv_layers = []
        conv_layers.append(approxNN.AdaPT_Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,padding=1, stride=1, axx_mult=axx_mult_global))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))
                
        conv_layers.append(nn.ReLU())
                
        if max_pool_size > 1:
            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size))
        
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

    def only_output(self, x):
        fwd = self.layers(x)
        return self.output(fwd)

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

        if approx_linear:
            fc_layers.append(approxNN.AdaPT_Linear(input_size, output_size, axx_mult=axx_mult_global))
        else:
            fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))        
        self.layers = nn.Sequential(*fc_layers)

        if add_output:
            if approx_linear:
                self.output = approxNN.AdaPT_Linear(output_size, num_classes, axx_mult=axx_mult_global)
            else:
                self.output = nn.Linear(output_size, num_classes)
            self.no_output = False
        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd)

    def only_output(self, x):
        fwd = self.layers(x)
        return self.output(fwd)

    def only_forward(self, x):
        return self.layers(x), 0, None

class VGG_SDN(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(VGG_SDN, self).__init__()
        # read necessary parameters
        self.input_size = 32
        self.num_classes = num_classes
        self.conv_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512] # the first element is input dimension
        self.fc_layer_sizes = [512, 512]

        # read or assign defaults to the rest
        self.max_pool_sizes = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
        self.conv_batch_norm = True
        self.init_weights = init_weights
        self.add_output = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs

        self.num_output = sum(self.add_output) + 1

        self.init_conv = nn.Sequential() # just for compatibility with other models
        self.layers = nn.ModuleList()
        self.init_depth = 0
        self.end_depth = 2

        # add conv layers
        input_channel = 3
        cur_input_size = self.input_size
        output_id = 0
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size/2)
            conv_params =  (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            add_output = self.add_output[layer_id]
            output_params = (add_output, self.num_classes, cur_input_size, output_id)
            self.layers.append(ConvBlockWOutput(conv_params, output_params))
            input_channel = channel
            output_id += add_output
        
        fc_input_size = cur_input_size*cur_input_size*self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            add_output = self.add_output[layer_id + len(self.conv_channels)]
            output_params = (add_output, self.num_classes, output_id)
            self.layers.append(FcBlockWOutput(fc_params, output_params, flatten=flatten))
            fc_input_size = width
            output_id += add_output
        
        end_layers = []
        if approx_linear:
            end_layers.append(approxNN.AdaPT_Linear(fc_input_size, self.fc_layer_sizes[-1], axx_mult=axx_mult_global))
        else:
            end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        
        end_layers.append(nn.Dropout(0.5))
        if approx_linear:
            end_layers.append(approxNN.AdaPT_Linear(self.fc_layer_sizes[-1], self.num_classes, axx_mult=axx_mult_global))
        else:
            end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))

        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, approxNN.AdaPT_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, approxNN.AdaPT_Linear) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
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
                
                # uncertainty = -(1*torch.logsumexp(output  / 1, dim=1)).mean().item()
                uncertainty = -(1*torch.logsumexp(output  / 1, dim=1)).item()
                uncertainties.append(uncertainty)
                # print(self.uncertainty_threshold,self.confidence_threshold)
            
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
        
        # uncertainty = -(1*torch.logsumexp(output  / 1, dim=1)).mean().item()
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
        

def _vgg(arch, pretrained, progress, device, dataset_name, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False

        script_dir = os.path.dirname(__file__)
        script_dir= script_dir.rsplit('/', 1)[0]

        if dataset_name == "CIFAR10":
            kwargs["num_classes"] = 10
            state_dict = torch.load(script_dir + "/state_dicts/" + arch + ".pt", map_location=device)

        elif dataset_name == "CIFAR100":
            kwargs["num_classes"] = 100
            state_dict = torch.load(script_dir + "/state_dicts/" + arch + "_" + dataset_name + ".pt", map_location=device)
        
        model = VGG_SDN(**kwargs)
        model.load_state_dict(state_dict)
    else:
        model = VGG_SDN(**kwargs)
    
    return model

def vgg16_sdn_bn(pretrained=False, progress=True, device="cpu", axx_mult = 'mul8s_acc', dataset_name="CIFAR10", **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """    
    global axx_mult_global
    axx_mult_global = axx_mult
    
    return _vgg("vgg16_sdn_bn", pretrained, progress, device, dataset_name, **kwargs)

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
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))