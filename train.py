#-*- coding:utf-8 -*-
# +
import os 
from datetime import datetime
import argparse
import json
import torch
from torchvision.transforms import Compose, Lambda
import numpy as np

import pandas as pd 
from sklearn.model_selection import train_test_split
from typing import Tuple

from diffusion_model.trainer import MRIQT, Trainer
from diffusion_model.unet import create_model
from diffusion_model.utils import Perceptual3DLoss
from feat_ext.feature_extractor import VGG3DFeatureExtractor
from data.dataset import UnpairedDataset, PairedDataset


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def load_csv_split(df, test_size=0.2, random_state=RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and validation sets based on subject IDs.
    Ensures that subjects with multiple entries are included only in the training set.
    Args:
        df (pd.DataFrame): DataFrame containing file paths and subject IDs.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
    Returns:
        train_df (pd.DataFrame): Training set DataFrame.
        val_df (pd.DataFrame): Validation set DataFrame.        
    """
    subject_counts = df['sub'].value_counts()
    duplicate_subjects = subject_counts[subject_counts > 1].index.tolist()
    unique_subjects = subject_counts[subject_counts == 1].index.tolist()
    train_subjects = set(duplicate_subjects)
    unique_train, unique_val = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state
    )

    train_subjects.update(unique_train)
    val_subjects = set(unique_val)

    train_df = df[df['sub'].isin(train_subjects)]
    val_df = df[df['sub'].isin(val_subjects)]

    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    return train_df, val_df

@torch.inference_mode()
def load_pretrained_perceptual_loss():
    """
    Load a pre-trained 3D VGG feature extractor and set up the perceptual loss function.
    Returns:
        Perceptual3DLoss: Configured perceptual loss function.
    """
    feat = VGG3DFeatureExtractor().cuda()
    state = torch.load('models/feature_extractor.pth', map_location='cuda', weights_only=True)
    feat.load_state_dict(state, strict=False)
    for param in feat.parameters():
        param.requires_grad = False
    feat.eval()
    layer_indices= (0, 1, 2, 3)
    layer_weights = [1., 1., 0.75, 0.5]
    perceptual_loss_fn = Perceptual3DLoss(feat, layer_indices, layer_weights).cuda()
    return perceptual_loss_fn

def train(args):

    # save the arguments within results_folder
    os.makedirs(args.results_folder, exist_ok=True)
    with open(os.path.join(args.results_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
        f.write('\n')

    transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: 2 * (t - t.min()) / (t.max() - t.min() + 1e-8) - 1),  # Scale to [-1, 1]
        Lambda(lambda t: t.unsqueeze(0)),  # from W, H, D to 1, W, H, D
        Lambda(lambda t: t.permute(0, 3, 2, 1)),  # from 1, W, H, D to 1, D, H, W
    ])

    if args.paired_training:
        print("Using Paired Training")
        hf_csv = '/path/to/paired_hf_dataset.csv'  # Update this path accordingly
        df = pd.read_csv(hf_csv)
        train_df, val_df = load_csv_split(df, test_size=0.2, random_state=RANDOM_SEED)
        
        dataset = PairedDataset(
            dataframe=train_df, 
            input_size=args.input_size,
            depth_size=args.depth_size,
            transform=transform,
        )

        val_dataset = PairedDataset(
            dataframe=val_df,
            input_size=args.input_size,
            depth_size=args.depth_size,
            transform=transform,
        )
    else:

        if os.path.exists('train_split.csv') and os.path.exists('val_split.csv'):
            print("Using existing train_split.csv and val_split.csv")
            train_df = pd.read_csv('train_split.csv')
            val_df = pd.read_csv('val_split.csv')
        else:

            hf_csv = '/path/to/hf_dataset.csv'  # Update this path accordingly

            df = pd.read_csv(hf_csv)
            train_df, val_df = load_csv_split(df, test_size=0.2, random_state=RANDOM_SEED)
            
        dataset = UnpairedDataset(
            dataframe=train_df, 
            input_size=args.input_size,
            depth_size=args.depth_size,
            transform=transform,
        )

        val_dataset = UnpairedDataset(
            dataframe=val_df,
            input_size=args.input_size,
            depth_size=args.depth_size,
            transform=transform,
        )

    print(f'Number of training samples: {len(dataset)}, Number of validation samples: {len(val_dataset)}\n{"#"*40}')

    perceptual_loss_fn = load_pretrained_perceptual_loss().cuda()

    unet = create_model(
        image_size=args.input_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions="20,10", # or "16,8" if input_size is 2^n
        in_channels=2,
        out_channels=1,
    ).cuda()

    mriqt = MRIQT(
        unet, 
        image_size = args.input_size,
        depth_size = args.depth_size,
        timesteps = args.timesteps,   
        loss_type = args.loss_type, # 'l1', 'l2', 'mixed'
        perceptual_loss_fn=perceptual_loss_fn,
        use_cfg=args.use_cfg, # True, # whether to use classifier free guidance
        cond_drop_prob=args.cond_drop_prob,
        guidance_weight=args.guidance_weight, 
        parametrization=args.parametrization, # 'v' or 'eps'
    ).cuda()

    if len(args.resume_weight) > 0:
        weight = torch.load(args.resume_weight, map_location='cuda')
        mriqt.load_state_dict(weight['ema'], strict=False)
        print("Model Loaded!")

    now = datetime.now()

    trainer = Trainer(
        mriqt,
        dataset,
        val_dataset = val_dataset,
        image_size = args.input_size,
        depth_size = args.depth_size,
        train_batch_size = args.batchsize,
        train_lr = args.train_lr,
        train_num_steps = args.epochs, 
        gradient_accumulate_every = args.gradient_accumulate_every, 
        eval_interval = args.eval_interval, 
        save_and_sample_every = args.save_and_sample_every,
        results_folder = args.results_folder,
        warmup=args.warmup, 
    )
    trainer.train()
    end_time = datetime.now()
    print(f'Training time: {end_time - now}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--input_size', type=int, default=160)
    parser.add_argument('--depth_size', type=int, default=160)
    parser.add_argument('--num_channels', type=int, default=64)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--train_lr', type=float, default=2e-5)
    parser.add_argument('--warmup', action='store_true', default=False) # number of steps for warmup if resuming
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--gradient_accumulate_every', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30001)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=250) # interval for evaluation and sampling
    parser.add_argument('--save_and_sample_every', type=int, default=500)
    parser.add_argument('--loss_type', type=str, default='mixed') # 'l1', 'l2', 'mixed'
    parser.add_argument('-r', '--resume_weight', type=str, default="")

    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--parametrization', type=str, default='v')
    parser.add_argument('--use_cfg', action='store_true', default=False) # Use classifier-free guidance
    parser.add_argument('--guidance_weight', type=float, default=2) # CFG guidance weight --> 1 is only condition guidance
    parser.add_argument('--use_self_conditioning', action='store_true', default=False) # Use self-conditioning
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    parser.add_argument('--lambda_perc', type=float, default=0.25) # weight for perceptual loss
    parser.add_argument('--paired_training', action='store_true', default=False) # Use paired training
   
    args = parser.parse_args()

    train(args)