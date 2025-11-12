import numpy as np 
import ants 
import pandas as pd 
import os 
from dataset import PairedDataset

def compute_tikhonov_transfer_function(hf_kspace_list, lf_kspace_list, lambda_reg=1e-3): 
    """ Estimate the transfer function S(f) using Tikhonov-regularized least squares. 
    Args: 
    hf_kspace_list: List of np.arrays (complex) - High-field images in k-space. 
    lf_kspace_list: List of np.arrays (complex) - Low-field images in k-space. 
    lambda_reg: float - Regularization strength to stabilize division. 
    Returns: S_f: 
    np.array (complex) - Estimated transfer function in k-space. """ 
    # Initialize numerator and denominator 
    numerator = np.zeros_like(hf_kspace_list[0], dtype=np.complex64) 
    denominator = np.zeros_like(hf_kspace_list[0], dtype=np.float32) 
    for hf_k, lf_k in zip(hf_kspace_list, lf_kspace_list): 
        numerator += lf_k * np.conj(hf_k) 
        denominator += np.abs(hf_k) ** 2 
        # Add regularization to avoid division by small values 
        S_f = numerator / (denominator + lambda_reg) 
    return S_f

csv_file = '/path/to/paired_hf_dataset.csv'  # Update this path accordingly
df = pd.read_csv(csv_file)

dataset = PairedDataset(
    dataframe=df, 
    input_size=160,
    depth_size=160,
    transform=None,  # No transform needed for k-space extraction
)

hf_kspace_list = []
lf_kspace_list = []
for i in range(len(dataset)):
    lf_img = dataset[i]['input']
    hf_img = dataset[i]['target']
    # Compute k-space using FFT
    lf_kspace = np.fft.fftn(lf_img, norm='ortho')
    hf_kspace = np.fft.fftn(hf_img, norm='ortho')
    lf_kspace_list.append(lf_kspace)
    hf_kspace_list.append(hf_kspace)

S_f = compute_tikhonov_transfer_function(hf_kspace_list, lf_kspace_list, lambda_reg=1e-3)
# Save the transfer function
np.save('transfer_function.npy', S_f)
print("Tikhonov transfer function saved to 'data/transfer_function.npy'")