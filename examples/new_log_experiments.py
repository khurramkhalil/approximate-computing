import os
import zipfile
import random
import numpy as np
import torch

import csv
from datetime import datetime

import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch.nn as nn

from util_func import get_dataset, prep_adapt_dataset

from models.SDNs.vgg_sdn import vgg16_sdn_bn
from models.SDNs.wideresnet_sdn import wideresnet_sdn_v1
from models.SDNs.mobilenet_sdn import mobilenet_sdn_v1
# import models.SDNs.fault_injection as fie
import models.SDNs.sdm_fault_mitigation as sdm

threads = 20
torch.set_num_threads(threads)

# maybe better performance
# Setting environment variables in Python
os.environ["OMP_PLACES"] = "cores"
os.environ["OMP_PROC_BIND"] = "close"
os.environ["OMP_WAIT_POLICY"] = "active"

def get_random_seed():
    return 1221 # 121 and 1221

def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())


# Function to save results to CSV
def save_result_to_csv(result, filename, file_exists=False):
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, result.keys())
        if not file_exists:
            dict_writer.writeheader()
        dict_writer.writerow(result)

# Generate CSV filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"joint_experiment_results_{timestamp}.csv"


# Define the models, approximate multipliers, and fault rates
models = [vgg16_sdn_bn, wideresnet_sdn_v1, mobilenet_sdn_v1]
approx_mults = ['mul8s_1KV6', 'mul8s_1KV8', 'mul8s_1KV9', 'mul8s_1KVP', 'mul8s_1L2J', 'mul8s_1L2H', 'mul8s_1L2N', 'mul8s_1L12']
fault_rates = [10, 30, 50]

# Define fault points for each model
fault_points = {
    0: ['layers.0.layers.0.weight'],
    1: ['init_conv.weight'],
    2: ['init_conv.0.weight']
}

# Set up constants
confidence_threshold = 0.5
uncertainty_threshold = 8

# Nested loops for experiments
for idx, model_class in enumerate(models):
    for axx_mult in approx_mults:
        for FR in fault_rates:
            print(f"Running experiment: Model: {model_class.__name__}, Approx Mult: {axx_mult}, Fault Rate: {FR}%")
            
            # Initialize model
            model = model_class(pretrained=True, axx_mult=axx_mult)
            model.eval()

            # Set random seeds and prepare datasets
            set_random_seeds()
            data_t, trainset_1 = prep_adapt_dataset()
            t_dataset = get_dataset()
            one_batch_dataset = get_dataset(1, False)



            from pytorch_quantization import nn as quant_nn
            from pytorch_quantization import calib
            import torch
            from tqdm import tqdm

            def collect_stats(model, data_loader, num_batches):
                """Feed data to the network and collect statistics."""
                
                # Enable calibrators
                for name, module in model.named_modules():
                    if isinstance(module, quant_nn.TensorQuantizer):
                        if module._calibrator is not None:
                            module.disable_quant()
                            module.enable_calib()
                        else:
                            module.disable()

                # Feed data and collect statistics
                for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
                    model(image.cpu())
                    if i >= num_batches:
                        break

                # Disable calibrators
                for name, module in model.named_modules():
                    if isinstance(module, quant_nn.TensorQuantizer):
                        if module._calibrator is not None:
                            module.enable_quant()
                            module.disable_calib()
                        else:
                            module.enable()

            def compute_amax(model, **kwargs):
                """Load calibration results and compute amax."""
                
                # Load calibration result
                for name, module in model.named_modules():
                    if isinstance(module, quant_nn.TensorQuantizer):
                        if module._calibrator is not None:
                            if isinstance(module._calibrator, calib.MaxCalibrator):
                                module.load_calib_amax()
                            else:
                                module.load_calib_amax(**kwargs)
                        print(f"{name:40}: {module}")
                model.cpu()

            # It is a bit slow since we collect histograms on CPU
            with torch.no_grad():
                stats = collect_stats(model, data_loader=data_t, num_batches=2)
                amax = compute_amax(model, method="percentile", percentile=99.99)
                
                # Optional: test different calibration methods
                # amax = compute_amax(model, method="mse")
                # amax = compute_amax(model, method="entropy")
            
            # Get fault points for the current model
            FP = fault_points[idx]
            # Introduce fault
            faulty_model, correction = sdm.introduce_fault_with_sdm(model, FR, None, FP)
            print("Is there any correction: ", correction)

            # Test with fault
            top1_acc, top5_acc, early_output_counts, non_conf_output_counts, conf_violation_counts, unc_viol_with_fault = \
                sdm.sdn_test_early_exits_sdm(faulty_model, one_batch_dataset.test_loader, confidence_threshold, uncertainty_threshold, "cpu")

            # Store results
            result = {
                'model': model_class.__name__,
                'approx_mult': axx_mult,
                'fault_rate': FR,
                'fault_points': ','.join(FP),
                'top1_acc': top1_acc,
                'top5_acc': top5_acc,
                'early_output_counts': ','.join(map(str, early_output_counts)),
                'non_conf_output_counts': ','.join(map(str, non_conf_output_counts)),
                'conf_violation_counts': ','.join(map(str, conf_violation_counts)),
                'unc_viol_with_fault': ','.join(map(str, unc_viol_with_fault))
            }

            # Save result to CSV
            save_result_to_csv(result, csv_filename, file_exists=os.path.exists(csv_filename))

            # Print results
            print("Results:", result)
            print(f"Results saved to {csv_filename}")
            print("-" * 80)

print(f"All experiments completed. Results saved to {csv_filename}")
