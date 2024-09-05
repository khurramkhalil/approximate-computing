import torch
import numpy as np
import random
from collections import deque

class DifferentialWeightSnapshots:
    def __init__(self, model, snapshot_frequency=100, max_snapshots=5, compression_ratio=0.1):
        self.model = model
        self.snapshot_frequency = snapshot_frequency
        self.max_snapshots = max_snapshots
        self.compression_ratio = compression_ratio
        self.initial_state = self.get_model_state()
        self.snapshots = deque(maxlen=max_snapshots)
        self.iteration = 0
        
    def get_model_state(self):
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def compress_differences(self, differences):
        compressed = {}
        for name, diff in differences.items():
            flat_diff = diff.flatten()
            k = int(len(flat_diff) * self.compression_ratio)
            values, indices = torch.topk(flat_diff.abs(), k)
            compressed[name] = (values, indices, flat_diff.shape)
        return compressed
    
    def decompress_differences(self, compressed):
        decompressed = {}
        for name, (values, indices, shape) in compressed.items():
            flat_diff = torch.zeros(shape.numel(), device=values.device)
            flat_diff[indices] = values
            decompressed[name] = flat_diff.reshape(shape)
        return decompressed
    
    def take_snapshot(self):
        current_state = self.get_model_state()
        differences = {name: (current_state[name] - self.initial_state[name]) for name in current_state}
        compressed_diff = self.compress_differences(differences)
        self.snapshots.append(compressed_diff)
    
    def restore_from_snapshot(self):
        if not self.snapshots:
            return
        
        latest_snapshot = self.snapshots[-1]
        decompressed_diff = self.decompress_differences(latest_snapshot)
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(self.initial_state[name] + decompressed_diff[name])
    
    def step(self):
        self.iteration += 1
        if self.iteration % self.snapshot_frequency == 0:
            self.take_snapshot()

def introduce_fault_with_dws(model, percent_of_faults, fault_loc=None, layer_to_attack=None, dws=None):
    model.eval()
    for name, param in model.named_parameters():
        if layer_to_attack == name:
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
            for i in range(0, len(fault_loc)):
                wf1[fault_loc[i]] = -wf1[fault_loc[i]]
            wf11 = wf1.reshape(w1.shape)
            param.data = wf11
    
    if dws:
        dws.restore_from_snapshot()
    
    return model

from models.SDNs import data


def cnn_test_uncertainty(model, loader, device='cpu', dws=None):
    model.eval()
    to_np = lambda x: x.data.cpu().numpy()

    preds = []
    
    top1 = data.AverageMeter()
    top5 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(b_y.view(1, -1).expand_as(pred))
            l1 = correct.tolist()[0]
            
            pred = pred.tolist()[0]
            b_y = b_y.tolist()

            for i in range(len(l1)):
                softmax = torch.nn.functional.softmax(output[i:i+1][0], dim=0)
                p = to_np(softmax)
                logp = np.log2(p)
                entropy = np.sum(-p*logp)

                confidence = torch.max(softmax).item()
                uncertainty = -(1*torch.logsumexp(output[i:i+1]  / 1, dim=1)).item()
                if (l1[i]==True): 
                    preds.append([True, confidence, uncertainty, entropy])
                else:
                    preds.append([False, confidence, uncertainty, entropy])

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, preds

def sdn_test_uncertainty(model, loader, device='cpu', dws=None):
    model.eval()
    to_np = lambda x: x.data.cpu().numpy()
    _score_r = []
    _score_w = []
    preds = {}
    
    top1 = []
    top5 = []
    for output_id in range(model.num_output):
        t1 = data.AverageMeter()
        t5 = data.AverageMeter()
        top1.append(t1)
        top5.append(t5)
        preds[output_id] = []

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)

            for output_id in range(model.num_output):
                cur_output = output[output_id]
                prec1, prec5 = data.accuracy(cur_output, b_y, topk=(1, 5))
                top1[output_id].update(prec1[0], b_x.size(0))
                top5[output_id].update(prec5[0], b_x.size(0))
                #added
                _, pred = cur_output.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(b_y.view(1, -1).expand_as(pred))
                l1 = correct.tolist()[0]
                for i in range(len(l1)):
                    softmax = torch.nn.functional.softmax(cur_output[i:i+1][0], dim=0)
                
                    confidence = torch.max(softmax).cpu()
                    uncertainty = -(1*torch.logsumexp(cur_output[i:i+1]  / 1, dim=1)).item()
                    if (l1[i]==True): 
                        preds[output_id].append([True, uncertainty, confidence])
                    else:
                        preds[output_id].append([False, uncertainty, confidence])


    top1_accs = []
    top5_accs = []

    for output_id in range(model.num_output):
        top1_accs.append(top1[output_id].avg.data.cpu().numpy()[()])
        top5_accs.append(top5[output_id].avg.data.cpu().numpy()[()])

    return top1_accs, top5_accs, preds

def sdn_test_early_exits(model, loader, confidence_threshold = 0.5, uncertainty_threshold = None, device='cpu'):

    model.forward = model.early_exit
    model.confidence_threshold = confidence_threshold
    model.uncertainty_threshold = uncertainty_threshold

    model.eval()
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output
    conf_violation_counts = [0] * model.num_output
    unc_violation_counts = [0] * model.num_output

    top1 = data.AverageMeter()
    top5 = data.AverageMeter()

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
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

    return top1_acc, top5_acc, early_output_counts, non_conf_output_counts, conf_violation_counts, unc_violation_counts
