import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np

from utils import c2r

class modl_dataset(Dataset):
    def __init__(self, mode, dataset_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.prefix = 'trn' if mode == 'train' else 'tst'
        self.dataset_path = dataset_path
        self.sigma = sigma

    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (2 x nrow x ncol) - float32
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """
        with h5.File(self.dataset_path, 'r') as f:
            gt, mask = f[self.prefix+'Org'][index], f[self.prefix+'Mask'][index]
        x0 = undersample(gt, mask, self.sigma)
        print("x0: type=", type(x0), "shape=", x0.shape, "dtype=", x0.dtype)
        print("gt: type=", type(gt), "shape=", gt.shape, "dtype=", gt.dtype)
        # mask = np.stack([mask, mask])
        # return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(c2r(csm)), torch.from_numpy(mask)
                # Ensure correct data types
        x0 = (x0 - np.min(x0)) / (np.max(x0) - np.min(x0) + 1e-8)  # Avoid division by zero
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-8)  # Normalize ground truth
    
        return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(mask), gt

    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Mask'])
        return num_data
"""
def undersample(gt, mask, sigma):
    ncoil, nrow, ncol = csm.shape
    sample_idx = np.where(mask.flatten()!=0)[0]
    noise = np.random.randn(len(sample_idx)*ncoil) + 1j*np.random.randn(len(sample_idx)*ncoil)
    noise = noise * (sigma / np.sqrt(2.))
    b = piA(gt, csm, mask, nrow, ncol, ncoil) + noise #forward model
    atb = piAt(b, csm, mask, nrow, ncol, ncoil)
    return atb
"""
def undersample(gt, mask, sigma=0.01):
    """
    Perform undersampling without coil sensitivity maps (csm).
    
    Args:
        gt: Ground truth image (2D or 3D array).
        mask: Undersampling mask (same spatial dimensions as `gt`).
        sigma: Noise standard deviation (not used in this version).
    
    Returns:
        atb: Undersampled and noisy k-space data.
    """
    nrow, ncol = mask.shape
    sample_idx = np.where(mask.flatten() != 0)[0]
    noise = np.random.randn(len(sample_idx)) + 1j*np.random.randn(len(sample_idx))
    noise = noise * (sigma / np.sqrt(2.))
    # Forward model (undersample the Fourier transform of `gt`)
    
    # Compute the k-space (Fourier transform) of the ground truth image
    gt_fft = np.fft.fft2(gt)
    
    # Shift the zero-frequency component to the center
    gt_fft_shifted = np.fft.fftshift(gt_fft)

    # Create a new k-space array to store the undersampled k-space
    gt_fft_sampled = np.zeros_like(gt_fft_shifted, dtype=complex)

    # Apply the undersampling mask
    gt_fft_sampled[np.unravel_index(sample_idx, (nrow, ncol))] = gt_fft_shifted[np.unravel_index(sample_idx, (nrow, ncol))]+noise

    # Shift the zero-frequency component back to its original location before inverse FFT
    gt_fft_sampled_shifted_back = np.fft.ifftshift(gt_fft_sampled)

    # Inverse FFT to obtain the undersampled image
    atb = np.fft.ifft2(gt_fft_sampled_shifted_back)
    
    return atb


def piA(im, csm, mask, nrow, ncol, ncoil):
    """
    fully-sampled image -> undersampled k-space
    """
    im = np.reshape(im, (nrow, ncol))
    im_coil = np.tile(im, [ncoil, 1, 1]) * csm #split coil images
    k_full = np.fft.fft2(im_coil, norm='ortho') #fft
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    k_u = k_full[mask!=0]
    return k_u

def piAt(b, csm, mask, nrow, ncol, ncoil):
    """
    k-space -> zero-filled reconstruction
    """
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    zero_filled = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
    zero_filled[mask!=0] = b #zero-filling
    img = np.fft.ifft2(zero_filled, norm='ortho') #ifft
    coil_combine = np.sum(img*csm.conj(), axis=0).astype(np.complex64) #coil combine
    return coil_combine