import os
import torch
import lpips
import numpy as np
from contextlib import redirect_stdout, redirect_stderr, nullcontext
import os 
from monai.metrics import SSIMMetric, MultiScaleSSIMMetric
from scipy.stats import pearsonr 
from skimage.metrics import peak_signal_noise_ratio


def standardize_volume(data):
    """
    Accepts NIfTI image, numpy array, or torch tensor.
    Returns numpy array of shape [160,160,160] for single volume.
    If batch shape [B, 1, 160,160,160], returns [160,160,160] for first item.
    """
    # NIfTI image
    if hasattr(data, 'get_fdata'):
        arr = data.get_fdata()
    # NIfTI image ants
    elif hasattr(data, 'numpy'):
        arr = data.numpy()
    # Torch tensor
    elif isinstance(data, torch.Tensor):
        if len(data.shape) == 5 and data.shape[1] == 1:
            arr = data[0, 0].detach().cpu().numpy()
        elif len(data.shape) == 4 and data.shape[0] == 1:
            arr = data[0].detach().cpu().numpy()
        else:
            arr = data.detach().cpu().numpy()
    # Numpy array
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    # Remove batch and channel dims if present
    while arr.ndim > 3:
        arr = arr[0]
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")
    return arr


def compute_lpips(
    GEN, GT, axis=None, lpips_model=None, device="cuda",
    brain_mask=None, min_fg_ratio=0.10,
    normalize="zscore_slice"  # "zscore_slice" | "zscore_volume" | "minmax_volume"
):
    """
    Compute LPIPS between two 3D volumes slice-by-slice.
    Args:
        
        GEN, GT: torch.Tensor or np.ndarray [D,H,W]
        axis: 0 coronal, 1 sagittal, 2 axial; if None -> mean of all 3
        brain_mask: optional [D,H,W] (bool/0-1), same orientation as volumes
        min_fg_ratio: minimum foreground ratio in slice to include in LPIPS computation
        normalize: normalization method before LPIPS computation
    Returns:
        float: LPIPS value
    """
    # --- setup
    if isinstance(GEN, np.ndarray): GEN = torch.from_numpy(GEN)
    if isinstance(GT,  np.ndarray): GT  = torch.from_numpy(GT)
    GEN = GEN.float().to(device)
    GT  = GT.float().to(device)
    if brain_mask is not None:
        if isinstance(brain_mask, np.ndarray): brain_mask = torch.from_numpy(brain_mask)
        brain_mask = brain_mask.float().to(device)

    assert GEN.shape == GT.shape, "GEN and GT must have same shape"
    D, H, W = GT.shape

    if lpips_model is None:
        lpips_model = lpips.LPIPS(net="vgg")
    lpips_model = lpips_model.to(device).eval()

    def _norm(x2d, mode):
        if mode == "zscore_slice":
            m, s = x2d.mean(), x2d.std()
            x = (x2d - m) / (s + 1e-8)
            x = x / (x.abs().max() + 1e-8)  # to [-1,1]-ish without clipping
        elif mode == "zscore_volume":
            # caller should pass pre-zscored volumes for best speed; fallback here:
            x = x2d
            m, s = x.mean(), x.std()
            x = (x - m) / (s + 1e-8)
            x = x / (x.abs().max() + 1e-8)
        elif mode == "minmax_volume":
            mn, mx = x2d.min(), x2d.max()
            x = 2 * (x2d - mn) / (mx - mn + 1e-8) - 1
        else:
            raise ValueError("normalize must be zscore_slice|zscore_volume|minmax_volume")
        return x.clamp(-1, 1)

    def _lpips_on_axis(a, b, msk, ax):
        # get number of slices
        ns = a.shape[ax]
        batch_a, batch_b = [], []
        for i in range(ns):
            if ax == 0:
                sA, sB = a[i], b[i]
                sM = None if msk is None else msk[i]
            elif ax == 1:
                sA, sB = a[:, i, :], b[:, i, :]
                sM = None if msk is None else msk[:, i, :]
            else:
                sA, sB = a[:, :, i], b[:, :, i]
                sM = None if msk is None else msk[:, :, i]

            if sM is not None:
                fg_ratio = (sM > 0.5).float().mean().item()
                if fg_ratio < min_fg_ratio:
                    continue
                sA = sA * sM
                sB = sB * sM

            sA = _norm(sA, normalize)
            sB = _norm(sB, normalize)

            # make [1,3,H,W]
            sA = sA.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)
            sB = sB.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)
            batch_a.append(sA)
            batch_b.append(sB)

        if not batch_a:
            return float("nan")

        x = torch.cat(batch_a, dim=0)  # [N,3,H,W]
        y = torch.cat(batch_b, dim=0)

        with torch.no_grad():
            vals = lpips_model(x, y).view(-1)  # [N]
        return float(vals.mean().item())

    if axis is None:
        vals = [_lpips_on_axis(GEN, GT, brain_mask, ax) for ax in (0,1,2)]
        return float(np.nanmean(vals))
    else:
        return _lpips_on_axis(GEN, GT, brain_mask, axis)

def compute_mse(GEN, GT, mask):
    """Mean Squared Error (efficient, NumPy)."""
    GEN = (GEN - GEN.min()) / (GEN.max() - GEN.min() + 1e-8)
    GT = (GT - GT.min()) / (GT.max() - GT.min() + 1e-8)
    mask = standardize_volume(mask)
    return np.mean((GT[mask] - GEN[mask]) ** 2)

def compute_mae(GEN, GT, mask):
    """Mean Absolute Error (efficient, NumPy)."""
    GEN = standardize_volume(GEN)
    GT = standardize_volume(GT)
    mask = standardize_volume(mask)
    GEN = (GEN - GEN.min()) / (GEN.max() - GEN.min() + 1e-8)
    GT = (GT - GT.min()) / (GT.max() - GT.min() + 1e-8)
    return np.mean(np.abs(GT[mask] - GEN[mask]))

def compute_rmse(GEN, GT, mask):
    """Root Mean Squared Error (efficient, NumPy)."""
    GEN = standardize_volume(GEN)
    GT = standardize_volume(GT)
    mask = standardize_volume(mask)
    GEN = (GEN - GEN.min()) / (GEN.max() - GEN.min() + 1e-8)
    GT = (GT - GT.min()) / (GT.max() - GT.min() + 1e-8)
    return np.sqrt(compute_mse(GEN, GT, mask))

def compute_psnr_skimage(GEN, GT, mask):
    """Peak Signal-to-Noise Ratio (skimage)."""
    GEN = standardize_volume(GEN)
    GT = standardize_volume(GT)
    mask = standardize_volume(mask)
    GEN = (GEN - GEN.min()) / (GEN.max() - GEN.min() + 1e-8)
    GT = (GT - GT.min()) / (GT.max() - GT.min() + 1e-8)
    return peak_signal_noise_ratio(GT, GEN, data_range=1.0)

def compute_pearson(GEN, GT, mask):
    """Pearson correlation (efficient, NumPy)."""
    GEN = standardize_volume(GEN)
    GT = standardize_volume(GT)

    mask = standardize_volume(mask)
    GEN = (GEN - GEN.min()) / (GEN.max() - GEN.min() + 1e-8)
    GT = (GT - GT.min()) / (GT.max() - GT.min() + 1e-8)
    return np.corrcoef(GEN[mask].flatten(), GT[mask].flatten())[0, 1]
    # return np.corrcoef(GT, GEN)[0, 1]

def compute_ssim_monai(GEN, GT, mask):
    """Structural Similarity Index Measure (MONAI)."""

    GEN_tensor = torch.from_numpy(GEN).unsqueeze(0).unsqueeze(0).float()
    GT_tensor = torch.from_numpy(GT).unsqueeze(0).unsqueeze(0).float()
    ssim_metric = SSIMMetric(spatial_dims=3)#, win_size=7)

    return ssim_metric(GT_tensor, GEN_tensor).item()

def compute_multiscale_ssim_monai(GEN, GT, mask):
    """Multiscale Structural Similarity Index Measure (MONAI)."""

    GEN_tensor = torch.from_numpy(GEN).unsqueeze(0).unsqueeze(0).float()
    GT_tensor = torch.from_numpy(GT).unsqueeze(0).unsqueeze(0).float()
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, weights=[0.6, 0.2, 0.2], kernel_size=7)
    return ms_ssim(GEN_tensor, GT_tensor).item() 

def compute_mlc(GEN):
    """Computes the mean local correlation (MLC) of a volume."""

    GEN = standardize_volume(GEN)
    GEN = (GEN - GEN.min()) / (GEN.max() - GEN.min() + 1e-8)

    correlations = []

    h, w, d = GEN.shape

    def is_constant_or_nan(x):
        # peak-to-peak robustly detects constant arrays; skip NaNs too
        if np.isnan(x).any():
            return True
        return np.isclose(np.ptp(x), 0.0)

    # Correlations along axis 0
    for i in range(h - 1):
        slice1 = GEN[i, :, :].ravel()
        slice2 = GEN[i + 1, :, :].ravel()
        if is_constant_or_nan(slice1) or is_constant_or_nan(slice2):
            continue
        corr, _ = pearsonr(slice1, slice2)
        correlations.append(corr)
        
    # Correlations along axis 1
    for j in range(w - 1):
        slice1 = GEN[:, j, :].ravel()
        slice2 = GEN[:, j + 1, :].ravel()
        if is_constant_or_nan(slice1) or is_constant_or_nan(slice2):
            continue
        corr, _ = pearsonr(slice1, slice2)
        correlations.append(corr)
    
    # Correlations along axis 2
    for k in range(d - 1):
        slice1 = GEN[:, :, k].ravel()
        slice2 = GEN[:, :, k + 1].ravel()
        if is_constant_or_nan(slice1) or is_constant_or_nan(slice2):
            continue
        corr, _ = pearsonr(slice1, slice2)
        correlations.append(corr)

    return float(np.mean(correlations)) if correlations else 0.0

def compute_mslc(GEN):
    """
    Mean Shifted Line Correlation (MSLC) - Average Nyquist Ghosting
    Range: [0, 1] (lower is better - detects ghosting artifacts)
    Works for 2D and 3D images
    """
    GEN = standardize_volume(GEN)
    GEN = (GEN - GEN.min()) / (GEN.max() - GEN.min() + 1e-8)

    correlations = []

    h, w, d = GEN.shape
    half_h = h // 2
    half_w = w // 2
    half_d = d // 2

    def is_constant_or_nan(x):
        # peak-to-peak robustly detects constant arrays; skip NaNs too
        if np.isnan(x).any():
            return True
        return np.isclose(np.ptp(x), 0.0)

    # Correlations along axis 0 at half-distance
    for i in range(half_h):
        slice1 = GEN[i, :, :].flatten()
        slice2 = GEN[i + half_h, :, :].flatten()
        if is_constant_or_nan(slice1) or is_constant_or_nan(slice2):
            continue
        if np.std(slice1) > 0 and np.std(slice2) > 0:
            corr, _ = pearsonr(slice1, slice2)
            correlations.append(corr)
    
    # Correlations along axis 1 at half-distance
    for j in range(half_w):
        slice1 = GEN[:, j, :].flatten()
        slice2 = GEN[:, j + half_w, :].flatten()
        if is_constant_or_nan(slice1) or is_constant_or_nan(slice2):
            continue
        if np.std(slice1) > 0 and np.std(slice2) > 0:
            corr, _ = pearsonr(slice1, slice2)
            correlations.append(corr)
    
    # Correlations along axis 2 at half-distance
    for k in range(half_d):
        slice1 = GEN[:, :, k].flatten()
        slice2 = GEN[:, :, k + half_d].flatten()
        if is_constant_or_nan(slice1) or is_constant_or_nan(slice2):
            continue
        if np.std(slice1) > 0 and np.std(slice2) > 0:
            corr, _ = pearsonr(slice1, slice2)
            correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0

def compute_metrics(GEN, GT, mask=None, device='cuda', lpips_model=None, lpips_axis=None):
    """
    Computes a suite of image quality metrics between generated and ground truth volumes.
    This version collects variants from multiple libraries so you can directly compare them.
    """
    GEN = standardize_volume(GEN)
    GT = standardize_volume(GT)
    GEN = (GEN - GEN.min()) / (GEN.max() - GEN.min() + 1e-8)
    GT = (GT - GT.min()) / (GT.max() - GT.min() + 1e-8)
    # ensure default LPIPS model available
    if lpips_model is None:
        try:
            lpips_model = get_lpips(net='alex', device=device, quiet=True)
        except Exception:
            lpips_model = None

    # default mask: GT>0
    mask = GT > 0 if mask is None else mask

    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"error: {e}"

    metrics = {}

    metrics['LPIPS'] = _safe(compute_lpips, GEN, GT, axis=lpips_axis, lpips_model=lpips_model, device=device)

    metrics['MSE'] = _safe(compute_mse, GEN, GT, mask)
    metrics['MAE'] = _safe(compute_mae, GEN, GT, mask)
    metrics['RMSE'] = _safe(compute_rmse, GEN, GT, mask)

    metrics['PSNR'] = _safe(compute_psnr_skimage, GEN, GT, mask)

    metrics['Pearson'] = _safe(compute_pearson, GEN, GT, mask)

    metrics['SSIM'] = _safe(compute_ssim_monai, GEN, GT, mask)

    metrics['MS-SSIM'] = _safe(compute_multiscale_ssim_monai, GEN, GT, mask)

    metrics['MLC'] = _safe(compute_mlc, GEN)
    metrics['MSLC'] = _safe(compute_mslc, GEN)

    return metrics

_lpips_singleton = {}
def get_lpips(net='alex', device='cpu', quiet=True):
    """
    Return a singleton LPIPS model for given net/device.
    quiet=True suppresses LPIPS stdout/stderr during weight loading.
    """
    key = f"{net}-{device}"
    if key in _lpips_singleton:
        return _lpips_singleton[key]
    with open(os.devnull, "w") as fnull:
        ctx = redirect_stdout(fnull) if quiet else nullcontext()
        ctx2 = redirect_stderr(fnull) if quiet else nullcontext()
        with ctx, ctx2:
            model = lpips.LPIPS(net=net)
    model = model.to(torch.device(device))
    _lpips_singleton[key] = model
    return model
