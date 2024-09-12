import torch
import random
import numpy as np
from tqdm import tqdm
from scipy import stats
from models.SDNs import data
from adapt.approx_layers import axx_layers as approxNN

import pdb
# class StatisticalDistributionMonitoring:
#     def __init__(self, model, sample_size=100, significance_level=0.05, check_frequency=100):
#         self.model = model
#         self.sample_size = sample_size
#         self.significance_level = significance_level
#         self.check_frequency = check_frequency
#         self.initial_stats = self.compute_initial_stats()
#         self.iteration = 0

#     def compute_initial_stats(self):
#         stats = {}
#         for name, param in self.model.named_parameters():
#             if param.dim() > 1:  # Only consider weights, not biases
#                 weights = param.data.cpu().numpy().flatten()
#                 stats[name] = {
#                     'mean': np.mean(weights),
#                     'std': np.std(weights),
#                     'sample': np.random.choice(weights, size=self.sample_size, replace=False)
#                 }
#         return stats

#     def check_distribution(self):
#         for name, param in self.model.named_parameters():
#             if name in self.initial_stats:
#                 current_weights = param.data.cpu().numpy().flatten()
#                 current_sample = np.random.choice(current_weights, size=self.sample_size, replace=False)
                
#                 ks_statistic, p_value = stats.ks_2samp(self.initial_stats[name]['sample'], current_sample)
                
#                 if p_value < self.significance_level:
#                     return name  # Return the name of the affected layer
        
#         return None  # No significant changes detected

#     def step(self):
#         self.iteration += 1
#         if self.iteration % self.check_frequency == 0:
#             return self.check_distribution()
#         return None

# def targeted_fault_mitigation(model, layer_name):
#     with torch.no_grad():
#         param = dict(model.named_parameters())[layer_name]
#         mean = param.data.mean()
#         std = param.data.std()
#         # Replace extreme outliers with values drawn from a normal distribution
#         mask = torch.abs(param.data - mean) > 3 * std
#         param.data[mask] = torch.normal(mean, std, size=param.data[mask].shape)

class StatisticalDistributionMonitoring:
    def __init__(self, model,significance_level=0.10, check_frequency=100):
        self.model = model
        self.significance_level = significance_level
        self.check_frequency = check_frequency
        self.initial_stats = self.compute_initial_stats()
        self.iteration = 0

    def compute_initial_stats(self):
        stats = {}
        for name, parameter in self.model.named_parameters():
            # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, approxNN.AdaPT_Conv2d) or isinstance(module, approxNN.AdaPT_Linear):
            if 'weight' in name:
                weights = parameter.data.cpu().numpy().flatten()

                stats[name] = {
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'sample': np.random.choice(weights, size=len(weights), replace=False)
                }
        return stats

    def check_layer_distribution(self, layer_name):
        if layer_name not in self.initial_stats:
            return False

        parameter = dict(self.model.named_parameters())[layer_name]
        current_weights = parameter.data.cpu().numpy().flatten()

        current_sample = np.random.choice(current_weights, size=len(current_weights), replace=False)
        
        result = stats.anderson_ksamp([self.initial_stats[layer_name]['sample'], current_sample])
        
        return result.significance_level < self.significance_level

    def check_distributions(self):
        affected_layers = []
        for name in self.initial_stats.keys():
            if self.check_layer_distribution(name):
                affected_layers.append(name)
        return affected_layers

    def step(self):
        self.iteration += 1
        if self.iteration % self.check_frequency == 0:
            return self.check_distributions()
        return []

def targeted_fault_mitigation(model, layer_name):
    parameter = dict(model.named_parameters())[layer_name]
    with torch.no_grad():
        mean = parameter.data.mean()
        std = parameter.data.std()
        # Replace extreme outliers with values drawn from a normal distribution
        mask = torch.abs(parameter.data - mean) > 3 * std
        # parameter.data[mask] = torch.normal(mean, std, size=parameter.data[mask].shape)
        parameter.data[mask] = 0

def sdn_test_early_exits_sdm(model, loader, confidence_threshold=0.5, uncertainty_threshold=None, device='cpu'):
    model.forward = model.early_exit
    # model.forward = model.early_exit_only
    model.confidence_threshold = confidence_threshold
    model.uncertainty_threshold = uncertainty_threshold

    # sdm = StatisticalDistributionMonitoring(model)

    model.eval()
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output
    conf_violation_counts = [0] * model.num_output
    unc_violation_counts = [0] * model.num_output

    top1 = data.AverageMeter()
    top5 = data.AverageMeter()

    fault_mitigations = 0

    with torch.no_grad():
        # Wrap the loader with tqdm for progress bar
        for batch in tqdm(loader, desc="Testing", leave=False):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            
            # # Check for distribution changes and apply fault mitigation if needed
            # affected_layer = sdm.step()
            # if affected_layer:
            #     targeted_fault_mitigation(model, affected_layer)
            #     fault_mitigations += 1
            
            output, output_id, is_early, violations = model(b_x)

            if is_early:
                early_output_counts[output_id] += 1
            else:
                non_conf_output_counts[output_id] += 1
            
            for i in violations:
                if i[1] == 'unc':
                    unc_violation_counts[i[0]] += 1
                else:
                    conf_violation_counts[i[0]] += 1

            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))

            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, early_output_counts, non_conf_output_counts, conf_violation_counts, unc_violation_counts #, fault_mitigations

# Function to introduce faults and use SDM
def introduce_fault_with_sdm(model, percent_of_faults, fault_loc=None, layer_to_attack=None):
    sdm = StatisticalDistributionMonitoring(model)
    model.eval()
    for name, param in model.named_parameters():
        if name in layer_to_attack: 
            print("Attacked layer", name)
            print(param.shape)
            w1 = param.data
            wf1 = torch.flatten(w1)
            no_of_faults = int(percent_of_faults * len(wf1) / 100)
            if no_of_faults > len(wf1):
                no_of_faults = len(wf1)

            print("Number of weights attacked", no_of_faults)
            if fault_loc is None:
                fault_loc = random.sample(range(0, len(wf1)), no_of_faults)
                fault = [random.uniform(-2, 2) for _ in range(len(fault_loc))]
            for i in range(0, len(fault_loc)):
                # wf1[fault_loc[i]] = -wf1[fault_loc[i]]
                wf1[fault_loc[i]] = torch.tensor(fault[i])
            wf11 = wf1.reshape(w1.shape)
            param.data = wf11
    
    affected_layers = sdm.check_distributions()
    correction = 0
    if affected_layers:
        for affected_layer in affected_layers:
            targeted_fault_mitigation(model, affected_layer)
            correction += 1
    
    return model, correction