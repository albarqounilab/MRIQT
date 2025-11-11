#-*- coding:utf-8 -*-
from torch.utils.data import Dataset
import ants
import torchio as tio
import numpy as np
import pandas as pd
from skimage.exposure import match_histograms

class UnpairedDataset(Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 input_size=160,
                 depth_size=160,
                 transform=None,
                 res='hf',
                 ): 
        """
        Dataset for unpaired high-field MRI images used for training diffusion models.
        Args:
            dataframe (pd.DataFrame): DataFrame containing file paths and subject IDs.
            input_size (int): Target size for width and height.
            depth_size (int): Target size for depth.
            transform (callable, optional): Optional transform to be applied on a sample.
            res (str): Resolution type, default is 'hf' for high-field.
                Use 'lf' for low-field only for inference.
        """
        self.df = dataframe 
        self.input_size = input_size
        self.depth_size = depth_size
        self.transform = transform
        self.res = res 
        self.transfer_function = np.load('assets/transfer_function.npy')
        try:
            self.avg_lf = np.load('data/avg_lf.npy')
        except:
            print("Average low-field image not found. Histogram matching will be skipped.")
            self.avg_lf = None

    def read_image(self, file_path):
        """ 
        Read NIfTI image using ANTs and return as a NumPy array.
        Args:
            file_path (str): Path to the NIfTI file.
        Returns:
            np.ndarray: Image data as a NumPy array.
        """
        img = ants.image_read(file_path)
        img = ants.reorient_image2(img, 'RAS')
        img = img.numpy()
        return img
    
    def resize_img(self, img):
        """
        Resize image to target size using cropping or padding.
        Target size is (input_size, input_size, depth_size).
        Args:
            img (np.ndarray): Input image array.
        Returns:
            np.ndarray: Resized image array.
        """
        w, h, d = img.shape
        target_shape = (self.input_size, self.input_size, self.depth_size)
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...], affine=np.eye(4), dtype=img.dtype)
            cop = tio.CropOrPad(target_shape, padding_mode='reflect') 
            img = np.asarray(cop(img))[0]
        return img

    def degrade_hq(self, data):
        """
        Degrade high-quality image to low-quality.
        Args:
            data (np.ndarray): High-quality image data.
        Returns:
            np.ndarray: Degraded low-quality image data.
        """
        # Convert to k-space and normalize
        hq_k = np.fft.fftn(data, norm='ortho')
        mean_k, std_k = np.mean(hq_k), np.std(hq_k) + 1e-8
        hq_k = (hq_k - mean_k) / std_k # Normalize in k-space
        hq_k = hq_k.astype(np.complex64)
        
        # Apply transfer function to simulate low-field degradation
        degraded_k = hq_k * self.transfer_function
        degraded_k[0, 0, 0] = 0  # Zero the DC term
        degraded_k = degraded_k * std_k + mean_k # Denormalization in k-space
        degraded_img = np.fft.ifftn(degraded_k, norm='ortho').real # Back to image space

        # Post-process degraded image
        degraded_img = degraded_img - np.mean(degraded_img) # correct for intensity shift
        degraded_img = np.clip(degraded_img, 0, np.percentile(degraded_img, 99)) # Clip for correction
        degraded_img = (degraded_img - np.min(degraded_img)) / (np.max(degraded_img) - np.min(degraded_img) + 1e-8)

        # Final post-processing: histogram matching (optional)
        if self.avg_lf is not None:
            avg_lf = self.avg_lf # Average low-field image for histogram matching
            degraded_img = match_histograms(degraded_img, avg_lf)
            degraded_img = (degraded_img - np.min(degraded_img)) / (np.max(degraded_img) - np.min(degraded_img) + 1e-8)
            
        return degraded_img

    @staticmethod
    def prepare_img(img):
        """ Normalize and clip image intensities based on foreground mask. """
        mask = img > 0.01
        img = np.clip(img, np.percentile(img[mask], 1), np.percentile(img[mask], 99))
        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        file_path, subject_id = self.df[f'{self.res}'].iloc[idx], self.df['sub'].iloc[idx]

        target = self.read_image(file_path)
        target = self.resize_img(target)
        target = self.prepare_img(target)
        if self.res == 'hf':
            degraded = self.degrade_hq(target)
        else:
            degraded = target.copy()  # No degradation for other resolutions
                # e.g., if res='lf', just use the image as is --> will be discarded as a condition in training
            target = None  # No target available for low-field only
        if self.transform is not None:
            target = self.transform(target)
            degraded = self.transform(degraded)

        return {'input': degraded, 'target': target, 'subject_id': subject_id}

class PairedDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 input_size: int = 160, 
                 depth_size: int = 160, 
                 transform=None,
                 ):
        """
        Dataset for paired ultra-low-field and high-field MRI images.
        Args:
            dataframe (pd.DataFrame): DataFrame containing file paths and subject IDs.
            input_size (int): Target size for width and height.
            depth_size (int): Target size for depth.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.df = dataframe
        self.df = self.df.dropna()
        self.pair_files = self.get_pair_files()  # List of tuples (ulf, hf, sub)

        self.input_size = input_size
        self.depth_size = depth_size
        self.transform = transform
        
    def get_pair_files(self):
        """Reads the CSV and returns a list of tuples (ulf, hf, sub)."""
        pairs = []
        # sub,hf,lf
        for i, row in self.df.iterrows():
            hf = row['hf']
            ulf = row['lf']
            sub_id = row['sub']
            pairs.append({'hf': hf, 'ulf': ulf, 'sid': sub_id})    
        return pairs
     
    def read_image(self, file_path):
        """
        Read NIfTI image using ANTs and return as a NumPy array.
        Args:
            file_path (str): Path to the NIfTI file.
        Returns:
            np.ndarray: Image data as a NumPy array.
        """
        img = ants.image_read(file_path)
        img = ants.reorient_image2(img, 'RAS').numpy()
        return img
    
    def resize_img(self, img):
        """ 
        Resize image to target size using cropping or padding.
        Target size is (input_size, input_size, depth_size).
        Args:
            img (np.ndarray): Input image array.
        Returns:
            np.ndarray: Resized image array.
        """
        w, h, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...], affine=np.eye(4), dtype=img.dtype)
            cop = tio.CropOrPad((self.input_size, self.input_size, self.depth_size), padding_mode='reflect')  # Crop or pad to target shape
            img = np.asarray(cop(img))[0]
        return img

    @staticmethod
    def prepare_img(img):
        """ Normalize and clip image intensities based on foreground mask. """
        mask = img > 0.01
        img = np.clip(img, np.percentile(img[mask], 1), np.percentile(img[mask], 99))
        return img

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):

        hf_file = self.pair_files[index]['hf']
        ulf_file = self.pair_files[index]['ulf']
        
        ulf = self.read_image(ulf_file)
        ulf = self.resize_img(ulf)
        ulf = self.prepare_img(ulf)
        hf = self.read_image(hf_file)
        hf = self.resize_img(hf)
        hf = self.prepare_img(hf)

        if self.transform is not None:
            ulf = self.transform(ulf)
            hf = self.transform(hf)

        return {'input': ulf, 'target': hf, 'subject_id': self.pair_files[index]['sid']}
    