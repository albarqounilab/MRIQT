import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from glob import glob
import os
from sklearn.model_selection import KFold
import wandb 
import matplotlib.pyplot as plt
import math
import argparse 

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from .feature_extractor import VGG3DAutoencoder

class NiftiMRIDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        img = nib.load(self.file_list[idx]).get_fdata()
        img = np.expand_dims(img, axis=0)  # [1, 160, 160, 160]
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)  # z-score normalization
        return torch.from_numpy(img).float()


def kfold_autoencoder_training(file_list, k=5, batch_size=16, epochs=20, lr=1e-3, device='cuda'):
    """
    Train a 3D VGG-like autoencoder using K-Fold cross-validation.
    Args:
        file_list (list): List of file paths to NIfTI images.
        k (int): Number of folds for cross-validation.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train each fold.
        lr (float): Learning rate.
        device (str): Device to use ('cuda' or 'cpu').
    Returns:
        results (list): List of dictionaries containing metrics for each fold.
        trained_model (nn.Module): The trained autoencoder model from the last fold.
    """
    results = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(file_list)):
        print(f"\n--- Fold {fold+1}/{k} ---")
        train_files = [file_list[i] for i in train_idx]
        val_files = [file_list[i] for i in val_idx]
        train_dataset = NiftiMRIDataset(train_files)
        val_dataset = NiftiMRIDataset(val_files)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = VGG3DAutoencoder()
        # Use fold number in wandb run name
        wandb.init(project="vgg3d_autoencoder_mri", name=f"fold_{fold+1}", config={"epochs": epochs, "lr": lr, "fold": fold+1})
        trained_model, history = train_vgg3d_autoencoder(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
        wandb.finish()
        # Store final metrics for this fold
        results.append({
            "fold": fold+1,
            "best_val_loss": min(history['val_loss']),
            "best_val_psnr": max(history['val_psnr']),
            "best_val_ssim": max(history['val_ssim'])
        })
    # Print summary
    print("\nK-Fold Results:")
    for r in results:
        print(f"Fold {r['fold']}: Best Val Loss={r['best_val_loss']:.4f}, PSNR={r['best_val_psnr']:.2f}, SSIM={r['best_val_ssim']:.3f}")
    # Average metrics
    avg_loss = np.mean([r['best_val_loss'] for r in results])
    avg_psnr = np.mean([r['best_val_psnr'] for r in results])
    avg_ssim = np.mean([r['best_val_ssim'] for r in results])
    print(f"\nAverage across folds: Val Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.3f}")
    return results, trained_model


def train_vgg3d_autoencoder(model, dataloader, val_dataloader, epochs=10, lr=1e-3, device='cuda'):
    """
    Train the 3D VGG-like autoencoder.
    Args:
        model (nn.Module): The autoencoder model to train.
        dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        device (str): Device to use ('cuda' or 'cpu').
    Returns:
        model (nn.Module): The trained autoencoder model.
        history (dict): Training history containing losses and metrics.
    """
    wandb.init(project="vgg3d_autoencoder_mri", config={"epochs": epochs, "lr": lr})
    # ========================================
    # 1. Setup
    # ========================================
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.L1Loss()  # Autoencoder reconstruction loss (L1 preserves edges)
    
    def get_foreground_mask(x, threshold=0.05):
        # Simple mask: foreground = voxels above threshold (after normalization)
        return (x.abs() > threshold).float()
    # ========================================
    # 2. Training loop
    # ========================================
    best_val_loss = math.inf
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            out = model(x)
            tgt = F.interpolate(x, size=out.shape[2:], mode='trilinear', align_corners=False)
            # Focus loss on foreground only
            mask = get_foreground_mask(tgt)
            loss = criterion(out * mask, tgt * mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(dataloader)
        history['train_loss'].append(train_loss)
        wandb.log({"train_loss": train_loss, "epoch": epoch})
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

        # Validation every epoch
        model.eval()
        val_loss = 0.0
        val_psnr = []
        val_ssim = []
        with torch.no_grad():
            for x in val_dataloader:
                x = x.to(device)
                out = model(x)
                tgt = F.interpolate(x, size=out.shape[2:], mode='trilinear', align_corners=False)
                mask = get_foreground_mask(tgt)
                loss = criterion(out * mask, tgt * mask)
                val_loss += loss.item()
                # Metrics: PSNR, SSIM (on central slice, foreground only)
                out_np = (out[0,0] * mask[0,0]).cpu().numpy()
                tgt_np = (tgt[0,0] * mask[0,0]).cpu().numpy()
                mid = out_np.shape[0] // 2
                val_psnr.append(psnr(tgt_np, out_np, data_range=out_np.max()-out_np.min()))
                val_ssim.append(ssim(tgt_np[mid], out_np[mid], data_range=out_np[mid].max()-out_np[mid].min()))
                # Feature map visualization every 5 epochs (first batch only)
                if epoch % 5 == 0 and x.shape[0] > 0:
                    feats = model.encoder(x)
                    fig, axes = plt.subplots(1, len(feats), figsize=(4 * len(feats), 4))
                    if len(feats) == 1:
                        axes = [axes]
                    for i, (fmap, ax) in enumerate(zip(feats, axes)):
                        ax.imshow(fmap[0, 0, fmap.shape[2] // 2].cpu().numpy(), cmap='viridis')
                        ax.set_title(f'Feature map {i}, channel 0')
                        ax.axis('off')
                    plt.tight_layout()
                    wandb.log({f"feature_maps_epoch_{epoch}": plt})
                    plt.close(fig)
                    break  # Only visualize first batch
        val_loss = val_loss / len(val_dataloader)
        mean_psnr = np.mean(val_psnr)
        mean_ssim = np.mean(val_ssim)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(mean_psnr)
        history['val_ssim'].append(mean_ssim)
        wandb.log({"val_loss": val_loss, "val_psnr": mean_psnr, "val_ssim": mean_ssim, "epoch": epoch})
        print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Val PSNR: {mean_psnr:.2f}, Val SSIM: {mean_ssim:.3f}")
        scheduler.step(val_loss)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "ebest_vgg3d_autoencoder.pth")
    print("Training complete. Best val loss:", best_val_loss)
    wandb.finish()
    return model, history

def visualize_fmaps_side_by_side(feature_extractor, lf, hf, title_x="HF Input", title_y="LF Input"):
    feature_maps = {}
    def hook_fn(module, input, output, name):
        feature_maps[name] = output
    # Register hooks
    for name, layer in feature_extractor.features.named_children():
        if isinstance(layer, nn.LeakyReLU):
            layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
    # Forward pass
    with torch.no_grad():
        _ = feature_extractor(lf)
        fmaps_x = {k: v for k, v in feature_maps.items()}
        feature_maps.clear()
        _ = feature_extractor(hf)
        fmaps_y = {k: v for k, v in feature_maps.items()}
    # Plot side by side
    num_maps = len(fmaps_x)
    plt.figure(figsize=(6 * num_maps, 6))
    for i, (lname, fmap_x) in enumerate(fmaps_x.items()):
        fmap_y = fmaps_y[lname]
        slice_idx_x = fmap_x.shape[2] // 2
        slice_idx_y = fmap_y.shape[2] // 2
        # HF
        plt.subplot(2, num_maps, i + 1)
        plt.imshow(fmap_x[0, 0, slice_idx_x].cpu().numpy(), cmap='viridis')
        plt.title(f"{title_x}: {lname}")
        plt.axis('off')
        # LF
        plt.subplot(2, num_maps, num_maps + i + 1)
        plt.imshow(fmap_y[0, 0, slice_idx_y].cpu().numpy(), cmap='viridis')
        plt.title(f"{title_y}: {lname}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('feature_maps_comparison.svg')
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_dir', help='Directory containing high-frequency images', type=str, required=True)
    args = parser.parse_args()
    file_list = glob(os.path.join(args.hf_dir, 'sub-*/anat/*T1w*gz'))
    print(f"Running k-fold cross-validation on {len(file_list)} images.")
    results, trained_model = kfold_autoencoder_training(file_list, k=5, batch_size=16, epochs=30, lr=1e-3, device='cuda')
