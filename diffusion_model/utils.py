import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================
# Perceptual 3D Loss Implementation
# ==============================================
def zscore_per_layer(feats, eps=1e-6):
    out = []
    for f in feats:
        mean = f.mean(dim=(2,3,4), keepdim=True)
        std  = f.std(dim=(2,3,4), keepdim=True)
        out.append((f - mean) / (std + eps))
    return out

class Perceptual3DLoss(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        layer_indices=(0,1,2,3),           
        layer_weights=None, 
        reduction="mean"
    ):
        super().__init__()
        self.feat = feature_extractor.eval()
        for p in self.feat.parameters():
            p.requires_grad = False          # freeze!

        self.layer_indices = tuple(layer_indices)
        if layer_weights is None:
            self.layer_weights = [1.0] * len(self.layer_indices)
        else:
            assert len(layer_weights) == len(self.layer_indices)
            self.layer_weights = list(layer_weights)

        self.reduction = reduction

    def forward(self, x_hat, x_ref, mask=None, step=None):
        """
        x_hat, x_ref: [B,1,D,H,W] or [B,1,H,W] — this version assumes 3D
            x_hat image to evaluate
            x_ref reference image
        mask:      [B,1,D,H,W] in {0,1} (optional, e.g., brain mask)
        """
        # ensure shape: [B,1,D,H,W]
        if x_hat.ndim == 4: x_hat = x_hat.unsqueeze(1)
        if x_ref.ndim == 4: x_ref = x_ref.unsqueeze(1)
        if x_hat.ndim == 3: x_hat = x_hat.unsqueeze(0).unsqueeze(0)
        if x_ref.ndim == 3: x_ref = x_ref.unsqueeze(0).unsqueeze(0)

        # per-sample standardization in image space (light)
        def zimg(t):
            m = t.mean(dim=(2,3,4), keepdim=True)
            s = t.std(dim=(2,3,4), keepdim=True)
            return (t - m) / (s + 1e-6)

        xh, xr = x_hat, x_ref
        
        # IMPORTANT: allow grads through x_hat path; keep ref path detached if desired
        feats_hat = self.feat(xh)  # grads w.r.t. xh propagate through frozen net
        with torch.no_grad():
            feats_ref = self.feat(xr)

        feats_hat = zscore_per_layer(feats_hat)
        feats_ref = zscore_per_layer(feats_ref)
        fig = None
        if step is not None and step % 100 == 0:
            fig = visualize_vgg3d_feature_difference(feats_hat, feats_ref, self.layer_indices)

        loss = x_hat.new_zeros(())
        for w, idx in zip(self.layer_weights, self.layer_indices):
            fh = feats_hat[idx]
            fr = feats_ref[idx]

            if fh.shape[2:] != fr.shape[2:]:
                fr = F.interpolate(fr, size=fh.shape[2:], mode="trilinear", align_corners=False)
            if mask is not None:
                m = F.interpolate(mask, size=fh.shape[2:], mode="nearest")
                l = (fh - fr).abs() * m
                denom = m.sum() * fh.shape[1] + 1e-6
                l = l.sum() / denom
            else:
                # l = (fh - fr).abs().mean()
                l = F.smooth_l1_loss(fh, fr)

            loss = loss + w * l

        if self.reduction == "mean":
            if fig is not None:
                return loss, fig
            else: 
                return loss
        return loss, fig

def visualize_vgg3d_feature_difference(feats_x, feats_y, layers_indices=None):


	num_layers = len(feats_x)
	z = feats_x[0].shape[-1] // 2
	if layers_indices is not None:
		num_layers = len(layers_indices)
		feats_x = [feats_x[i] for i in layers_indices]
		feats_y = [feats_y[i] for i in layers_indices]

	fig, axes = plt.subplots(3, num_layers, figsize=(5 * num_layers, 15))
	for i in range(num_layers):
		fx = feats_x[i].squeeze().detach().cpu().numpy()
		fy = feats_y[i].squeeze().detach().cpu().numpy()
		diff = abs(fx - fy)
		# print(fx.shape, fy.shape, diff.shape)
		axes[0, i].imshow(fx[0, :, :, fx.shape[-1]//2], cmap='viridis')
		axes[0, i].set_title(f'Recon Features Layer {i}')
		axes[1, i].imshow(fy[0, :, :, fy.shape[-1]//2], cmap='viridis')
		axes[1, i].set_title(f'GT Features Layer {i}')
		im = axes[2, i].imshow(diff[0, :, :, diff.shape[-1]//2], cmap='hot')
		axes[2, i].set_title(f'Diff Layer {i}')
		fig.colorbar(im, ax=axes[2, i])
	plt.tight_layout()
	plt.close() 
	return fig

# ==============================================
# Logging Helpers
# ==============================================


import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def visualize_sub(input, gen, target):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes[0, 0].imshow(input[input.shape[0] // 2, :, :].T, cmap='gray')
    axes[0, 1].imshow(input[:, input.shape[1] // 2, :].T, cmap='gray')
    axes[0, 2].imshow(input[:, :, input.shape[2] // 2].T, cmap='gray')
    axes[1, 0].imshow(gen[gen.shape[0] // 2, :, :].T, cmap='gray')
    axes[1, 1].imshow(gen[:, gen.shape[1] // 2, :].T, cmap='gray')
    axes[1, 2].imshow(gen[:, :, gen.shape[2] // 2].T, cmap='gray')
    axes[2, 0].imshow(target[target.shape[0] // 2, :, :].T, cmap='gray')
    axes[2, 1].imshow(target[:, target.shape[1] // 2, :].T, cmap='gray')
    axes[2, 2].imshow(target[:, :, target.shape[2] // 2].T, cmap='gray')
    plt.tight_layout()
    plt.close(fig)
    return fig



from tabulate import tabulate

def format(tensor):
    if tensor is None:
        return ("nan", "nan", "nan", "nan")
    tt = tensor.detach() if hasattr(tensor, "detach") else tensor
    return (f"{tt.max().item():.6f}", f"{tt.min().item():.6f}", f"{tt.mean().item():.6f}", f"{tt.std().item():.6f}")


def log_tensor_stats(
    step, pred, 
    x_start, x0_pred, condition_tensors, t, 
    noise=None, parametrization='eps',
    a_fn=None, s_fn=None
):
    """
    Print simple max/min/mean/std stats for several tensors when called.

    - step: current training step (logging occurs only if step is not None and step % interval == 0)
    - pred, x_start, x0_pred: required tensors
    - noise: optional tensor (used differently depending on `parametrization`)
    - parametrization: 'eps' or 'v' (if 'v', a_fn and s_fn must be provided)
    - t: optional time tensor passed to a_fn/s_fn when parametrization == 'v'
    - a_fn, s_fn: callables taking (t, shape) -> tensor, or precomputed tensors
    - condition_tensors: optional extra tensor to log
    """

    headers = ["Tensor", "max", "min", "mean", "std"]
    rows = []
    rows.append([f"predicted ({parametrization})", *format(pred)])

    if noise is not None:
        if parametrization == "eps":
            rows.append(["target noise/eps", *format(noise)])
        elif parametrization == "v":
            if a_fn is None or s_fn is None:
                raise ValueError("a_fn and s_fn must be provided for 'v' parametrization")
            a = a_fn(t, x_start.shape) if callable(a_fn) else a_fn
            s = s_fn(t, x_start.shape) if callable(s_fn) else s_fn
            v_target = a * noise - s * x_start
            rows.append(["target velocity/v", *format(v_target)])
        else:
            rows.append([f"target ({parametrization})", *format(noise)])

    rows.append(["reconstructed x0", *format(x0_pred)])
    rows.append(["original x0", *format(x_start)])

    if condition_tensors is not None:
        rows.append(["condition tensor", *format(condition_tensors)])

    print(f"Step {step}_{t.detach().cpu().item()} stats:")
    print(tabulate(rows, headers, tablefmt="github"))
    print(50 * "-")

def map_K_to_tstart(K=None, alpha_bar_target=None, alpha_bar_table=None):
    """
    Prefer alpha_bar matching if schedules differ; otherwise allow raw K.
    - K: int in [0, T-1] measured with the same schedule.
    - alpha_bar_target: float in (0,1), the desired cumulative SNR level.
    - alpha_bar_table: 1D tensor/list of length T with cumulative products.
    """
    if alpha_bar_target is not None:
        # pick the nearest noise level
        ab = torch.as_tensor(alpha_bar_table, dtype=torch.float32)
        idx = int((ab - alpha_bar_target).abs().argmin().item())
        return idx
    assert K is not None, "Provide K or alpha_bar_target."
    return int(K)
