import os 
import torch
from torchvision.transforms import Compose, Lambda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.dataset import PairedDataset
from feat_ext.feature_extractor import VGG3DFeatureExtractor
from diffusion_model.utils import Perceptual3DLoss
from diffusion_model.trainer import MRIQT


@torch.inference_mode()
def load_pretrained_perceptual_loss(device):
    """
    Load a pre-trained 3D VGG feature extractor and set up the perceptual loss function.
    Returns:
        Perceptual3DLoss: Configured perceptual loss function.
    """
    feat = VGG3DFeatureExtractor().to(device)
    state = torch.load('feat_ext/best_feat_ext.pth', map_location=device, weights_only=True)
    feat.load_state_dict(state, strict=False)
    for param in feat.parameters():
        param.requires_grad = False
    feat.eval()
    layer_indices= (0, 1, 2, 3)
    layer_weights = [1., 1., 0.75, 0.5]
    perceptual_loss_fn = Perceptual3DLoss(feat, layer_indices, layer_weights).to(device)
    return perceptual_loss_fn

def compute_avg_perceptual_loss(dataset, diffusion, perceptual_loss_fn, device):
    """
    Compute the average perceptual loss across the dataset at each timestep for each lf and hf.
    Returns:
        float: Average perceptual loss.
    """
    T = diffusion.num_timesteps
    hf_loss_dict = {t: [] for t in range(T)}    
    lf_loss_dict = {t: [] for t in range(T)}


    with torch.inference_mode():
        for i, sample in enumerate(dataset):
            lf = sample['lf'].unsqueeze(0).to(device)  # add batch dimension
            hf = sample['hf'].unsqueeze(0).to(device)
            sid = sample.get('subject_id', None)
            X_0 = hf.clone() 
            print(f'Processing subject {i+1}/{len(dataset)}: {sid}')

            for t in range(T):
                tt = torch.full((1,), t, dtype=torch.long).to(device)
                noisy_lf = diffusion.q_sample(x_start=lf, t=tt)
                noisy_hf = diffusion.q_sample(x_start=hf, t=tt)

                noisy_lf = (noisy_lf - noisy_lf.mean()) / (noisy_lf.std() + 1e-8)
                noisy_hf = (noisy_hf - noisy_hf.mean()) / (noisy_hf.std() + 1e-8)

                perc_loss_lf = perceptual_loss_fn(noisy_lf, X_0, step=1)
                perc_loss_hf = perceptual_loss_fn(noisy_hf, X_0, step=1)

                lf_loss_val = float(perc_loss_lf.detach().cpu().item())
                hf_loss_val = float(perc_loss_hf.detach().cpu().item())
                lf_loss_dict[t][sid] = lf_loss_val
                hf_loss_dict[t][sid] = hf_loss_val
    hf_loss_df = pd.DataFrame.from_dict(hf_loss_dict, orient='index')
    hf_loss_df.index.name = 't'
    lf_loss_df = pd.DataFrame.from_dict(lf_loss_dict, orient='index')
    lf_loss_df.index.name = 't'
    return hf_loss_df, lf_loss_df


# to smooth the curves
def moving_average(x, w=21):
    """ 
    Compute the moving average of a 1D array x with window size w.
    w: the number of consecutive samples the function averages for each output point
    smaller w, window size is small, less smoothing
    """
    if w <= 1: return x
    w = int(w) if int(w)%2==1 else int(w)+1
    pad = w//2
    xpad = np.pad(x, (pad, pad), mode='edge')
    ker = np.ones(w)/w
    return np.convolve(xpad, ker, mode='valid')

def plot_means(hf_loss_df, lf_loss_df):
    """
    Plot the mean perceptual loss for high-frequency (HF) and low-frequency (LF) components.
    Args:
        hf_loss_df (pd.DataFrame): DataFrame containing HF perceptual losses.
        lf_loss_df (pd.DataFrame): DataFrame containing LF perceptual losses.
    Returns:
        plt.Figure: Matplotlib figure object to be saved.
    """
    hf_lf_mean_diff = hf_loss_df.mean(axis=1) - lf_loss_df.mean(axis=1)
    hf_lf_std_diff = hf_loss_df.std(axis=1) - lf_loss_df.std(axis=1)
    hf_lf_mean_diff = moving_average(hf_lf_mean_diff.values, w=21)
    hf_lf_std_diff = moving_average(hf_lf_std_diff.values, w=21)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hf_lf_mean_diff, label='Mean HF - LF Perceptual Loss', color='blue')
    ax.fill_between(
        range(len(hf_lf_mean_diff)),
        hf_lf_mean_diff - hf_lf_std_diff,
        hf_lf_mean_diff + hf_lf_std_diff,
        color='blue', alpha=0.2,
        label='Std Dev'
    )
    ax.set_xlabel('Diffusion Timesteps')
    ax.set_ylabel('Perceptual Loss Difference')
    ax.set_title('Mean Perceptual Loss Difference between HF and LF Components')
    ax.legend()
    plt.close()
    return fig

def main(args, device):

    # Dataset 
    transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: 2 * (t - t.min()) / (t.max() - t.min() + 1e-8) - 1),  # Scale to [-1, 1]
        Lambda(lambda t: t.unsqueeze(0)),  # from W, H, D to 1, W, H, D
        Lambda(lambda t: t.permute(0, 3, 2, 1)),  # from 1, W, H, D to 1, D, H, W
    ])

    csv_file = args.csv_file
    df = pd.read_csv(csv_file)
    overall_dataset = PairedDataset(
        dataframe=df,
        input_size=160,
        depth_size=160,
        transform=transform,
    )

    diffusipn = MRIQT(
        denoise_fn=None,  # No Denoiser (UNet) is needed here
        image_size = 160,
        depth_size = 160,
        cond_drop_prob=0.1,
        guidance_weight=2.,
        use_cfg=True,
    ).to(device)

    perceptual_loss_fn = load_pretrained_perceptual_loss(device)

    hf_loss_df, lf_loss_df = compute_avg_perceptual_loss(
        overall_dataset, diffusipn, perceptual_loss_fn, device=device
    )
    fig = plot_means(hf_loss_df, lf_loss_df)
    fig.savefig(args.output_fig_path)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing dataset information.')
    parser.add_argument('--output_fig_path', type=str, required=True, help='Path to save the output figure.')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args, device)