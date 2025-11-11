#-*- coding:utf-8 -*-
# *Main part of the code is adopted from the following repository: https://github.com/mobaidoctor/med-ddpm
# Modifications have been made to adapt it to our image quality transfer task for MRI enhancement.

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from inspect import isfunction
from functools import partial
from pathlib import Path
import copy

import wandb
from tqdm import tqdm
import datetime
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from .utils import visualize_sub, log_tensor_stats
from .metrics import compute_metrics

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    # if fp16:
    #     with amp.scale_loss(loss, optimizer) as scaled_loss:
    #         scaled_loss.backward(**kwargs)
    # else:
    #     loss.backward(**kwargs)
    loss.backward(**kwargs)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        beta = self.beta
        one_minus_beta = 1.0 - beta
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data.mul_(beta).add_(current_params.data, alpha=one_minus_beta)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, dtype=torch.float32, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device, dtype=dtype).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device, dtype=dtype)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

class MRIQT(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        depth_size,
        channels=1,
        timesteps=1000,
        loss_type='l1',
        betas=None,

        perceptual_loss_fn=None,
        parametrization='v',
        use_cfg=True,
        guidance_weight=2., # CFG guidance weight -- 1.0 means no guidance (conditional only)
        cond_drop_prob=0.1,
        use_self_conditioning=False,
        i2i=True,
        use_percep_weighting=True, 
    ):
        """
        MRIQT Diffusion model for MRI image quality transfer.
        Args:
            denoise_fn (nn.Module): The denoising neural network model.
            image_size (int): The spatial size of the input images (assumed cubic).
            depth_size (int): The depth size of the input images.
            channels (int): Number of channels in the input images.
            timesteps (int): Number of diffusion timesteps.
            loss_type (str): Type of loss to use ('l1', 'l2', 'mixed').
            betas (np.ndarray, optional): Predefined beta schedule. If None, cosine schedule is used.
            perceptual_loss_fn (callable, optional): Perceptual loss function.
            parametrization (str): 'v' or 'eps' parametrization for diffusion.
            use_cfg (bool): Whether to use classifier-free guidance.
            guidance_weight (float): Weight for classifier-free guidance.
            cond_drop_prob (float): Probability of dropping the condition during training.
            use_self_conditioning (bool): Whether to use self-conditioning.
            i2i (bool): Whether to enable image-to-image translation mode.
            use_percep_weighting (bool): Whether to weight perceptual loss based on noise level.
        """

        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.perceptual_loss_fn = perceptual_loss_fn

        ### classifier free and v-parametrization  OCT 6
        self.parametrization = parametrization
        print(f"Running DDPM with {self.parametrization} parametrization")
        self.use_cfg = use_cfg
        self.guidance_weight = guidance_weight
        self.cond_drop_prob = cond_drop_prob
        self.use_self_conditioning = use_self_conditioning
        self.i2i = i2i
        self.use_percep_weighting = use_percep_weighting

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        
    # parameters for v formulation (the linear combination)
    def _a(self, t, shape):  return extract(self.sqrt_alphas_cumprod, t, shape) # sqrt(alpha_cumprod)
    def _s(self, t, shape):  return extract(self.sqrt_one_minus_alphas_cumprod, t, shape) # sqrt(1 - alpha_cumprod)

    def predict_x0_from_pred(self, x_t, t, pred, param='eps'):
        """
        Given the model prediction (either ε or v), compute the predicted x_0.
        Args:
            x_t: noised data at time t
            t: time step
            pred: model prediction (either ε or v)
            param: 'eps' or 'v' indicating the type of prediction
        Returns:
            predicted x_0
            """
        if param == 'eps': # ε-parametrization (original)
            return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred
        elif param == 'v':
            a = self._a(t, x_t.shape); s = self._s(t, x_t.shape)
            return a * x_t - s * pred
        else:
            raise ValueError(param)

    def predict_eps_from_pred(self, x_t, t, pred, param='eps'):
        """ 
        Given the model prediction (either ε or v), compute the predicted noise ε.
        Args:
            x_t: noised data at time t
            t: time step
            pred: model prediction (either ε or v)
            param: 'eps' or 'v' indicating the type of prediction
        Returns:
            predicted noise ε 
        """
        if param == 'eps': return pred
        elif param == 'v':
            a = self._a(t, x_t.shape); s = self._s(t, x_t.shape)
            return s * x_t + a * pred
        else:
            raise ValueError(param)

    def q_mean_variance(self, x_start, t):
        """
        Compute the mean and variance of q(x_t | x_0).
        Args:
            x_start: original data
            t: time step
        Returns:
            mean, variance, and log variance of q(x_t | x_0)
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the posterior q(x_{t-1} | x_t, x_0).
        Args:
            x_start: original data
            x_t: noised data at time t
            t: time step
        Returns:
            posterior mean, variance, and log variance
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, condition_tensors, clip_denoised: bool):
        """
        Compute the mean and variance of p(x_t | x_0, c).
        Args:
            x: noised data at time t
            t: time step
            condition_tensors: conditioning data
            clip_denoised: whether to clip the denoised data
        Returns:
        model mean, posterior variance, and log variance
        """
        if self.use_cfg: 
            p_c = self.denoise_fn(torch.cat([x, condition_tensors], dim=1), t)
            p_u = self.denoise_fn(torch.cat([x, torch.zeros_like(condition_tensors)], dim=1), t)
            pred = p_u + self.guidance_weight * (p_c - p_u)
        else:
            pred = self.denoise_fn(torch.cat([x, condition_tensors], dim=1), t) # Original prediction path

        x_recon = self.predict_x0_from_pred(x, t=t, pred=pred, param=self.parametrization) # predict x0 from either ε or v
        if clip_denoised: x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the distribution q(x_t | x_0).
        equation: x_t = sqrt(alphas_cumprod[t]) * x_start + sqrt(1 - alphas_cumprod[t]) * noise

        Args:
            x_start: original data
            t: time step
            noise: noise to be added
        Returns:
            noisy data at time t
        """
        noise = default(noise, lambda: torch.randn_like(x_start, dtype=x_start.dtype))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @torch.inference_mode()
    def p_sample(self, x, t, condition_tensors, clip_denoised=True, repeat_noise=False):
        """
        Sampling step for p(x_{t-1} | x_t, c) used during inference.
        Args:
            x: current data at time t
            t: current time step
            condition_tensors: conditioning data
            clip_denoised: whether to clip the denoised data
            repeat_noise: whether to use the same noise for all samples in the batch
        Returns:
            data at time t-1
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, condition_tensors=condition_tensors, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, x.dtype, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.inference_mode()
    def ddim_sample(self, x, t, condition_tensors, clip_denoised=True, eta=0.0):
        """
        DDIM sampling step with optional noise scale η for stochasticity why? To control diversity just like in DDPM.??? 
        Args:
            x: current data at time t
            t: current time step
            condition_tensors: conditioning data
            clip_denoised: whether to clip the denoised data
            eta: noise scale for control. 0.0 for deterministic DDIM, >0.0 for stochasticity
        Returns:
            data at time t-1
        """
        if self.use_cfg:
            p_c = self.denoise_fn(torch.cat([x, condition_tensors], dim=1), t) # conditioned prediction 
            p_u = self.denoise_fn(torch.cat([x, torch.zeros_like(condition_tensors)], dim=1), t) # unconditioned prediction 
            pred = p_u + self.guidance_weight * (p_c - p_u) # w condtioned + (w-1) unconditioned 
                # w = 1 --> only conditioned
                # w > 1 --> extrapolate away from unconditioned == stronger conditioning 
                    # w >> 1 --> risk of overfitting to condition and artifacts
                # w < 1 --> interpolate towards unconditioned == weaker conditioning 
                    # w ~ 0.5 --> very weak conditioning
                # w = 0 --> only unconditioned 
        else:
            pred = self.denoise_fn(torch.cat([x, condition_tensors], dim=1), t) # eps or v prediction path 

        x0_pred = self.predict_x0_from_pred(x, t, pred, param=self.parametrization) # either from noise or v
        if clip_denoised: x0_pred = x0_pred.clamp(-1, 1)

        a_t   = extract(self.alphas_cumprod, t, x.shape) # α(t) is α_cumprod at time t
        a_tm1 = extract(self.alphas_cumprod_prev, t, x.shape) # α(t-1) is α_cumprod at time t-1
        eps_hat = self.predict_eps_from_pred(x, t, pred, param=self.parametrization) # predicted noise from either ε or v

        ###
        sigma_sq = (eta ** 2) * ((1. - a_tm1) / (1. - a_t) * (1. - (a_t / a_tm1))).clamp_min(0.0)
        sigma_t = sigma_sq.sqrt()
        dir_coef = ((1. - a_tm1) - sigma_sq).clamp_min(0).sqrt()
        x_prev = a_tm1.sqrt() * x0_pred + dir_coef * eps_hat 
        if eta > 0.0:  
            x_prev = x_prev + sigma_t * torch.randn_like(x) 
        return x_prev


    def p_losses(self, x_start, t, condition_tensors=None, noise=None, step=None, results_folder=None): # OCT 6
        """
        Compute the loss for training the diffusion model.
        Args:
            x_start: original data
            t: time step
            condition_tensors: conditioning data (low-quality)
            noise: noise to be added
            step: current training step (for logging)
            results_folder: folder to save results (for logging)
        Returns: 
            Loss value
        """
        # ============================================================
        # 1. add noise to the data + conditioning 
        # ============================================================
        noise = default(noise, lambda: torch.randn_like(x_start, dtype=x_start.dtype))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # CFG training path: randomly drop condition
        if self.use_cfg and torch.rand(()) < self.cond_drop_prob:
            cond_in = torch.zeros_like(condition_tensors)
        else:
            cond_in = condition_tensors
        
        # ============================================================
        # self-conditioning
            # Change Unet in channels to in__channels + 1 CRITICAL - also in train.py when initializing model
        # ============================================================
        if self.use_self_conditioning and torch.rand(()) < 0.9:
            with torch.inference_mode():
                pred_tmp = self.denoise_fn(torch.cat([x_noisy, cond_in], dim=1), t)
                x0_prev = self.predict_x0_from_pred(x_noisy, t, pred_tmp, param=self.parametrization).detach()
            net_in = torch.cat([x_noisy, cond_in, x0_prev], dim=1)
        else:
            net_in = torch.cat([x_noisy, cond_in], dim=1)

        # ============================================================
        # 2. prediction
        # ============================================================
        pred = self.denoise_fn(net_in, t)  # predicts ε or v 

        # compute loss based on parametrization
        if self.parametrization == 'eps':

            loss_pred = F.l1_loss(pred, noise) # L1 loss on predicted noise

        elif self.parametrization == 'v':
            a = self._a(t, x_start.shape)
            s = self._s(t, x_start.shape) #a, s ??
            v_target = a * noise - s * x_start

            loss_pred = F.mse_loss(pred, v_target) # L2 loss on predicted v
        else:
            raise ValueError(self.parametrization)
        x0_pred = self.predict_x0_from_pred(x_noisy, t, pred, param=self.parametrization) # predicted x0 from either eps or v 

        # ============================================================
        # some logging
        # ============================================================
        if step is not None and step % 50 == 0:
            log_tensor_stats(step, pred, x_start, x0_pred, condition_tensors, t, noise, self.parametrization, a_fn=self._a(t, x_start.shape), s_fn=self._s(t, x_start.shape))

        # ============================================================
        # 3. image based losses and ablations
        # weighted perceptual loss based on noise level
        # ============================================================

        if self.perceptual_loss_fn is not None and self.loss_type == 'mixed':
            if self.use_percep_weighting:
                with torch.inference_mode():
                    a = self._a(t, x_start.shape); s = self._s(t, x_start.shape) + 1e-8
                    snr = (a * a) / (s * s)                       # ᾱ_t / (1-ᾱ_t)
                    _w_t = (snr / (1.0 + snr)).mean()              # map to [0,1], batch
                    if snr.mean() < 0.15:                          # hard skip if super noisy (~-15 dB)
                        return loss_pred, torch.zeros_like(loss_pred)
                w_t = _w_t.clone()
            perceptual_loss = self.perceptual_loss_fn(x0_pred, x_start, step=step)
            if isinstance(perceptual_loss, tuple):
                fig = perceptual_loss[1]; fig.savefig(f'{results_folder}/features{step}.svg')
                perceptual_loss = w_t * perceptual_loss[0]
        else:
            return loss_pred

        return loss_pred, perceptual_loss

    # @torch.no_grad() 
    @torch.inference_mode()
    def img2img(self, condition_tensors, strength=0.65, sampler="ddpm", clip_denoised=True, results_folder='./results'):
        """Image-to-image translation by adding noise to condition and denoising.
        Args: 
            condition_tensors: low-quality data, shape (B, C, D, H, W)
            strength: noise strength, between 0 and 1 --> K
            sampler: "ddpm" or "ddim"
            sampling_timesteps: number of DDIM steps (if using DDIM)
            eta: noise scale for DDIM (if using DDIM), between 0 and 1
        Returns:
            generated high-quality data, shape (B, C, D, H, W)
        """
        device = self.betas.device
        b, _, D, H, W = condition_tensors.shape
        t_start = int(strength * (self.num_timesteps - 1))
        t_tensor = torch.full((b,), t_start, device=device, dtype=torch.long)
        eps = torch.randn_like(condition_tensors) # noise
        x_t = self.q_sample(x_start=condition_tensors, t=t_tensor, noise=eps) # noised condition_tensors at t_start

        if sampler == 'ddpm':
            img = x_t
            for i in tqdm(reversed(range(0, t_start + 1)), total=t_start + 1, desc='img2img DDPM'): 
                t = torch.full((b,), i, device=device, dtype=torch.long) 
                img = self.p_sample(img, t, condition_tensors=condition_tensors, clip_denoised=clip_denoised) 
            return img

        elif sampler == "ddim":
            img = x_t  # must be x at t_start
            b = img.shape[0]

            times = torch.arange(t_start, -1, -1, device=device, dtype=torch.long)  # [t_start, ..., 0]
            for i in tqdm(times.tolist(), total=len(times), desc=f'img2img DDIM ({len(times)} steps)'):
                t = torch.full((b,), i, device=device, dtype=torch.long)
                img = self.ddim_sample(
                    img, t,
                    condition_tensors=condition_tensors,
                    clip_denoised=clip_denoised,
                    eta=0.0
                )
            return img
        else:
            raise ValueError("sampler must be 'ddpm' or 'ddim'")

    def forward(self, x, condition_tensors=None, *args, **kwargs):
        b, c, d, h, w, device, img_size, depth_size = *x.shape, x.device, self.image_size, self.depth_size
        assert h == img_size and w == img_size and d == depth_size, f'Expected dimensions: height={img_size}, width={img_size}, depth={depth_size}. Actual: height={h}, width={w}, depth={d}.'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, condition_tensors=condition_tensors, *args, **kwargs)

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        val_dataset=None,
        ema_decay=0.995,
        image_size=160,
        depth_size=160,
        train_batch_size=2,
        train_lr=2e-6,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        fp16=False,
        step_start_ema=1000,
        update_ema_every=10,
        eval_interval=1000,  # interval for evaluation and sampling
        save_and_sample_every=1000,
        results_folder='./results',
        warmup=True,
        strength=0.65, # full sampling (0.65 is our K)
        lambda_perc=0.25,
    ):
        """
        Trainer for MRIQT diffusion model.
        """
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()
        self.ema_model.to(next(self.model.parameters()).device)

        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.depth_size = depth_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.eval_interval = eval_interval

        self.ds = dataset
        self.val_dataset = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True))
        if self.val_dataset is not None:
            self.val_dl = cycle(data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True))
        else:
            self.val_dl = None

        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, weight_decay=2e-5)
        if warmup:
            warmup_scheduler = LinearLR(self.opt, start_factor=5e-6, total_iters=1000)
            cosine_scheduler = CosineAnnealingLR(self.opt, T_max=self.train_num_steps - 1000)
            self.scheduler = SequentialLR(self.opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[1000])
        else:
            self.scheduler = CosineAnnealingLR(self.opt, T_max=self.train_num_steps)

        #
        self.train_lr = train_lr
        self.train_batch_size = train_batch_size

        self.step = 0

        self.fp16 = fp16
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.log_dir = self.create_log_dir()
        self.strength = strength
        self.lambda_perc = lambda_perc

        project_name = str(self.results_folder)
        if '/' in project_name:
            project_name = project_name.replace('/', '_')
        wandb.init(
            project=project_name,  # Change this to your project name
            config={
                "lr": train_lr,
                "batchsize": train_batch_size,
                "image_size": image_size,
                "depth_size": depth_size,
                "train_num_steps": train_num_steps,
            }
        )
        self.reset_parameters()

    def create_log_dir(self):
        """
        Create a log directory for saving logs and checkpoints.
        Returns:
            log_dir (str): Path to the created log directory.
        """
        now = datetime.datetime.now().strftime(f"%y-%m-%dT%H%M%S_{self.results_folder}")
        log_dir = os.path.join("./logs", now)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def reset_parameters(self):
        """
        Reset model parameters and EMA model.
        """
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        """
        Update the EMA model.
        """
        if self.step == self.step_start_ema:
            self.ema_model.load_state_dict(self.model.state_dict())

        if self.step >= self.step_start_ema:
            self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'strategy': [self.model.parametrization, self.model.loss_type, self.model.use_cfg, self.model.cond_drop_prob],
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))   
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.opt.load_state_dict(data['optimizer'])
        for _ in range(self.step):
            self.scheduler.step()
        print(f'Loaded model from step {self.step}.')

    def _parse_loss(self, loss_output):
        """Parse loss output into components and compute total loss.
        Args:
            loss_output: output from the model's loss function, can be a single tensor or a tuple of tensors.
        Returns:
            total_loss: total loss tensor for backpropagation.
            pred_loss: prediction loss component (float).
            percep_loss: perceptual loss component (float) or None if not applicable.
        """
        if isinstance(loss_output, tuple) and len(loss_output) == 2:
            pred_loss, percep_loss = loss_output
            total_loss = 1.0 * pred_loss + self.lambda_perc * percep_loss
            return (
                total_loss.sum() / self.batch_size,
                pred_loss.item(),
                self.lambda_perc * percep_loss.item()
            )

        else:
            return (
                loss_output.sum() / self.batch_size,
                loss_output.item(),
                None
            )

    def _compute_grad_norm(self):
        """Compute L2 norm of gradients (returns float).
        Args:
            None
        Returns:
            total_norm (float): L2 norm of gradients.
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _gradient_clipping(self):
        """Clip gradients and log gradient norms.
        """
        # ============================================================
        # 1. compute pre-clipping grad norm
        # ============================================================
        pre_clip_norm = self._compute_grad_norm()
        wandb.log({'train/grad_norm_before_clipping': pre_clip_norm}, step=self.step)

        # ============================================================
        # 2. compute post-clipping grad norm
        # ============================================================
        post_clip_norm = self._compute_grad_norm()
        wandb.log({'train/grad_norm_after_clipping': post_clip_norm}, step=self.step)
        if 0.1 * self.train_num_steps >= 5000:
            clip_1, clip_2 = 1500, 3500
        elif self.train_num_steps <= 5501:
            clip_1, clip_2 = 1000, 1500
        else:
            clip_1, clip_2 = 1000, 2500

        if self.step < clip_1:
            clip_value = 0.3
        elif self.step < clip_2:
            clip_value = 0.4
        else:
            clip_value = 0.5

        # ============================================================
        # 3. gradient clipping
        # ============================================================
        if not hasattr(self, "grad_clip_calls"):
            self.grad_clip_calls = 0
        self.grad_clip_calls += 1
        wandb.log({'train/grad_clip_calls': self.grad_clip_calls}, step=self.step)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        
        # Compute gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        target_norm = 2.0
        if total_norm > target_norm:
            print(f'WARNING: Gradient norm {total_norm:.4f} exceeded target {target_norm:.4f} at step {self.step}, scaling down optimizer step.')
            for param_group in self.opt.param_groups:
                param_group['lr'] *= 0.95
            wandb.log({'train/lr_adjustment': 1, 'new_lr': param_group['lr']}, step=self.step)

        # Log gradient norm
        wandb.log({"train/grad_norm": total_norm}, step=self.step)

    def _log_validation_metrics(self, val_loss, val_pred, val_percep):
        """
        Log validation metrics.
        Args:
            val_loss (float): Total validation loss.
            val_pred (float): Prediction loss component.
            val_percep (float or None): Perceptual loss component.
        Returns:   
            None
        """
        log_msg = f'{self.step}: Val loss: {val_loss:.6f}, Pred: {val_pred:.6f}'
        if val_percep is not None:
            log_msg += f', Percep: {self.lambda_perc * val_percep:.6f}'
        print(log_msg)
        
        wandb_log = {"val/loss": val_loss, "val/pred_loss": val_pred}
        if val_percep is not None:
            wandb_log["val/perceptual_loss"] = self.lambda_perc * val_percep
        wandb.log(wandb_log, step=self.step)


    def _generate_and_evaluate_samples(self, val_input, val_target, subject_ids, 
                                    val_l1, val_percep, val_loss):
        """
        Generate samples and compute metrics.
        Args:
            val_input: low-quality input data for validation.
            val_target: high-quality target data for validation.
            subject_ids: list of subject IDs corresponding to the validation samples.
            val_l1: L1 loss on validation set.
            val_percep: Perceptual loss on validation set.
            val_loss: Total loss on validation set.
        Returns:
            None
        """
        self.ema_model.eval()
        milestone = self.step // self.save_and_sample_every
        
        # Generate images using img2img
        # with torch.no_grad():
        with torch.inference_mode():
            generated = self.ema_model.img2img(
                condition_tensors=val_input,
                strength=self.strength,
                sampler="ddpm",
                results_folder=str(self.results_folder),
            )
        
        # Convert to numpy
        gen_np = generated.permute(0, 1, 4, 3, 2).cpu().numpy() if len(generated.shape) == 5 \
                else generated.permute(0, 3, 2, 1).cpu().numpy()
        tgt_np = val_target.permute(0, 1, 4, 3, 2).cpu().numpy() if len(val_target.shape) == 5 \
                else val_target.permute(0, 3, 2, 1).cpu().numpy()
        inp_np = val_input.permute(0, 1, 4, 3, 2).cpu().numpy() if len(val_input.shape) == 5 \
                else val_input.permute(0, 3, 2, 1).cpu().numpy()
        
        # Print statistics
        print(f"{self.step}: Generated - max={gen_np.max():.4f}, min={gen_np.min():.4f}, "
            f"mean={gen_np.mean():.4f}, std={gen_np.std():.4f}")
        
        # Compute and save metrics
        self._compute_and_save_metrics(
            gen_np, tgt_np, inp_np, subject_ids
        )

        # Save model checkpoint
        self.save(milestone)
        
        print(f'Step {self.step} completed\n{"-"*100}')


    def _compute_and_save_metrics(self, gen_np, tgt_np, inp_np, subject_ids):
        """
        Compute metrics for generated samples and save results.
        Args:
            gen_np: Generated samples as a numpy array.
            tgt_np: Target samples as a numpy array.
            inp_np: Input samples as a numpy array.
            subject_ids: List of subject IDs corresponding to the samples.
        Returns:
            None
        """
        per_sub_metrics = {} 
        val_affine = np.eye(4)
        print(f"{20*'====='}\n")
        # ============================================================
        # 1. compute per-subject metrics and save visualizations + NIfTI files
        # ============================================================
        for i in range(gen_np.shape[0]):
            sample_img = gen_np[i][0]
            target_img = tgt_np[i][0]
            input_img = inp_np[i][0]

            # Compute metrics
            metrics_dict = compute_metrics(GEN=sample_img, GT=target_img)
            per_sub_metrics[subject_ids[i]] = metrics_dict
            try:
                print(f"{self.step}: {subject_ids[i]}:\n" + 
                    ", ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()]))
                print(f"{20*'====='}\n")
            except:
                print('Error printing metrics for subject:', subject_ids[i])
                continue
            # Visualize and save
            sub_fig = visualize_sub(input_img, sample_img, target_img)
            sub_fig.savefig(self.results_folder / f'step_{self.step}_gen_{subject_ids[i]}_comparison.svg')
            
            wandb.log({f'figures/{self.step}_{subject_ids[i]}': wandb.Image(sub_fig)}, step=self.step)
            
            # Save NIfTI
            nifti_img = nib.Nifti1Image(sample_img, affine=val_affine)
            nifti_img.to_filename(str(self.results_folder / f'step_{self.step}_{subject_ids[i]}_generated.nii.gz'))
                    
        print(f"{20*'====='}\n")
        
        averaged_metrics = {f"metrics/{n}": metrics_dict[n] 
                        if metrics_dict[n] else float('nan') 
                        for n in metrics_dict.keys()}
        wandb.log(averaged_metrics, step=self.step)
        
        # Save to CSV
        self._save_metrics_to_csv(
            subject_ids, metrics_dict
        )

    def _save_metrics_to_csv(self, subject_ids, metrics_dict):
            """Save metrics to CSV file."""
            csv_path = str(self.results_folder / f'subject_metrics.csv')
            new_rows = []
            for i, subj_id in enumerate(subject_ids):
                row = [
                    self.step, subj_id,
                ]
                
                for k, v in metrics_dict.items():
                    if isinstance(v, list):
                        val = metrics_dict[k][i] if i < len(metrics_dict[k]) else float('nan')
                    elif isinstance(v, np.ndarray):
                        val = metrics_dict[k][i] if i < v.shape[0] else float('nan')
                    elif isinstance(v, float) or isinstance(v, np.float32):
                        val = v
                    row.append(val)
                new_rows.append(row)

            columns = ['step', 'subject_id'] + list(metrics_dict.keys())

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                new_df = pd.DataFrame(new_rows, columns=columns)
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.DataFrame(new_rows, columns=columns)

            df.to_csv(csv_path, index=False)

    def _log_training_metrics(self, acc_loss, acc_pred, acc_percep):
        """Log training metrics to Weights & Biases."""
        avg_loss = np.mean(acc_loss)
        avg_pred = np.mean(acc_pred)
        current_lr = self.opt.param_groups[0]['lr']
        
        # Console logging
        # log_msg = f'{self.step}: Training loss: {avg_loss:.6f}, L1: {avg_l1:.6f}'
        log_msg = f'{self.step}: Training loss: {avg_loss:.6f}, Pred: {avg_pred:.6f}'
        if acc_percep:
            log_msg += f', Percep: {np.mean(acc_percep):.6f}'
        print(log_msg)
        
        wandb_log = {
            "learning_rate": current_lr,
            "train/loss": avg_loss,
            "train/pred_loss": avg_pred,
        }
        if acc_percep:
            wandb_log["train/perceptual_loss"] = np.mean(acc_percep)
        wandb.log(wandb_log, step=self.step)

    def _training_step(self, backwards):
        """Execute one training step with gradient accumulation."""
        torch.cuda.empty_cache()
        self.model.train()
        
        accumulated_loss = []
        accumulated_pred, accumulated_percep = [], []
        
        # Gradient accumulation loop
        for i in range(self.gradient_accumulate_every):
            data = next(self.dl)
            input_tensors = data['input'].cuda()
            target_tensors = data['target'].cuda()
            
            # Forward pass
            loss_output = self.model(
                target_tensors, 
                condition_tensors=input_tensors, 
                step=self.step, 
                results_folder=self.results_folder
            )
            
            # Parse loss components
            loss, pred_loss, percep_loss = self._parse_loss(loss_output)

            # Accumulate losses
            accumulated_loss.append(loss.item())
            accumulated_pred.append(pred_loss)
            if percep_loss is not None:
                accumulated_percep.append(percep_loss)
            
            # Backward pass
            backwards(loss / self.gradient_accumulate_every, self.opt)
        
        # Log training metrics
        self._log_training_metrics(
            # accumulated_loss, accumulated_l1, 
            accumulated_loss, accumulated_pred, 
            accumulated_percep
        )

    def _validation_and_sampling(self):
        """Run validation and generate samples."""
        self.model.eval()
        self.ema_model.eval()
        torch.cuda.empty_cache()
        
        # ============================================================
        # 1. VALIDATION LOSS
        # ============================================================
        try:
            if self.val_dl is None:
                raise RuntimeError("Validation DataLoader is not defined.")
            else:
                val_data = next(self.val_dl)
        except TypeError as e:
            raise RuntimeError(
                f"Error during validation data loading: {e}. "
                "Ensure validation dataset is properly defined."
            ) from e
        
        val_input = val_data['input'].cuda()
        val_target = val_data['target'].cuda()
        subject_ids = val_data['subject_id']
        
        assert val_input.shape == val_target.shape, \
            f"Shape mismatch: input {val_input.shape}, target {val_target.shape}"
        
        # with torch.no_grad():
        with torch.inference_mode():
            val_loss_output = self.model(
                val_target, 
                condition_tensors=val_input, 
                step=self.step, 
                results_folder=self.results_folder
            )
        
        # Parse and log validation loss
        val_loss, val_pred, val_percep = self._parse_loss(val_loss_output)
        self._log_validation_metrics(val_loss, val_pred, val_percep)

        # ============================================================
        # 2. SAMPLING & METRICS (every save_and_sample_every)
        # ============================================================
        if self.step % self.save_and_sample_every == 0:
            self._generate_and_evaluate_samples(
                val_input, val_target, subject_ids, 
                val_pred, val_percep, val_loss
            )

    def _finalize_training(self, start_time):
        """Finalize training and log final metrics."""
        print('Training completed')
        end_time = time.time()
        execution_time = (end_time - start_time) / 3600
        
        # Get final loss (need to store from last step)
        final_loss = 0.0  
        wandb.config.update({
            "execution_time (hour)": execution_time,
            "last_loss": final_loss
        })

    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()

        while self.step < self.train_num_steps: 
            self._training_step(backwards)
            
            self._gradient_clipping()
            
            self.opt.step()
            self.opt.zero_grad()
            self.scheduler.step()
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()
                # with torch.no_grad():
                with torch.inference_mode():
                    diff = 0.0
                    # for p, q in zip(self.model.parameters(), self.ema_model.parameters()):
                    #     diff += (p - q).abs().mean().item()
                    for p, q in zip(self.model.parameters(), self.ema_model.parameters()):
                        diff += (p.data - q.data).abs().mean().item()
                wandb.log({"train/ema_mean_abs_diff": diff}, step=self.step)

            if self.step % self.eval_interval == 0 and self.step != 0 and self.val_dl is not None:
                self._validation_and_sampling()

            self.step += 1
        self._finalize_training(start_time)

    # @torch.no_grad()
    @torch.inference_mode()
    def run_inference(
        self,
        split = "val",               # "train" or "val"
        paired = True,              # whether ground-truth targets are available
        sampler = "ddpm",            # "ddpm" or "ddim"
        max_batches = None,          # limit processed batches
    ):
        """
        Unified inference routine that properly reuses validation helper methods.
        Use this when you want to run inference after training.
        Args:
            split: which dataloader to use ("train" or "val")
            paired: whether targets exist for computing metrics
            sampler: "ddpm" or "ddim"
            max_batches: optional limit on number of batches
        """
        # ============================================================
        # 1) Select dataloader
        # ============================================================
        if split == "train":
            dl, dataset = self.dl, self.ds
        elif split == "val":
            dl, dataset = self.val_dl, self.val_dataset
        else:
            raise ValueError("split must be 'train' or 'val'")

        if dl is None or dataset is None:
            print(f"No dataset provided for split='{split}'.")
            return

        print(f"Starting inference on split='{split}' (paired={paired})...")
        total_batches = len(dataset) // self.train_batch_size
        if len(dataset) % self.train_batch_size != 0:
            total_batches += 1
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)

        progress_bar = tqdm(dl, desc="Inference", total=total_batches)

        for batch_idx, batch in enumerate(progress_bar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if batch_idx  == len(dataset):
                break

            self.ema_model.eval()
            torch.cuda.empty_cache()

            # Load batch
            val_input = batch['input'].cuda()
            subject_ids = batch.get('subject_id', [f"sub_{batch_idx}"])
            has_target = ('target' in batch) and paired
            val_target = batch['target'].cuda() if has_target else None

            # Validate shapes if paired
            if has_target and val_target is not None:
                assert val_input.shape == val_target.shape, \
                    f"Shape mismatch: input {val_input.shape}, target {val_target.shape}"

            # with torch.no_grad():
            with torch.inference_mode():
                generated = self.ema_model.img2img(
                    condition_tensors=val_input,
                    strength=self.strength,
                    sampler=sampler,
                    results_folder=self.results_folder,
                )

            # To numpy
            to_np = lambda x: (x.permute(0, 1, 4, 3, 2).cpu().numpy() if len(x.shape) == 5 
                              else x.permute(0, 3, 2, 1).cpu().numpy())
            gen_np = to_np(generated)
            inp_np = to_np(val_input)
            tgt_np = to_np(val_target) if has_target else None

            # Print stats
            print(f"{batch_idx}: Generated - max={gen_np.max():.4f}, min={gen_np.min():.4f}, "
                  f"mean={gen_np.mean():.4f}, std={gen_np.std():.4f}")
            print(f"{batch_idx}: Input - max={inp_np.max():.4f}, min={inp_np.min():.4f}, "
                  f"mean={inp_np.mean():.4f}, std={inp_np.std():.4f}")
            if has_target and tgt_np is not None:
                print(f"{batch_idx}: Target - max={tgt_np.max():.4f}, min={tgt_np.min():.4f}, "
                      f"mean={tgt_np.mean():.4f}, std={tgt_np.std():.4f}")

            val_l1 = val_percep = val_loss = None
            if has_target and tgt_np is not None:
                # Reuse _compute_and_save_metrics
            
                self._compute_and_save_metrics(
                    gen_np, tgt_np, inp_np, subject_ids,
                    val_l1, val_percep, val_loss
                )
            else:
                # Save without metrics (unpaired case)
                self._save_unpaired_results(gen_np, inp_np, subject_ids)

        print("Inference completed.")


    def _save_unpaired_results(self, gen_np, inp_np, subject_ids):
        """Save results when no ground truth is available (unpaired inference)."""
        # val_affine = np.array([[-1., 0., 0., 0.], [0., 1., 0., 0.],
        #                        [0., 0., -1., 0.], [0., 0., 0., 1.]])
        val_affine = np.eye(4)
        
        for i in range(gen_np.shape[0]):
            sample_img = gen_np[i][0]
            input_img = inp_np[i][0]

            # Simple 2x3 visualization
            sub_fig, axes = plt.subplots(2, 3, figsize=(10, 10))
            axes[0, 0].imshow(input_img[input_img.shape[0] // 2, :, :].T, cmap='gray')
            axes[0, 1].imshow(input_img[:, input_img.shape[1] // 2, :].T, cmap='gray')
            axes[0, 2].imshow(input_img[:, :, input_img.shape[2] // 2].T, cmap='gray')
            axes[1, 0].imshow(sample_img[sample_img.shape[0] // 2, :, :].T, cmap='gray')
            axes[1, 1].imshow(sample_img[:, sample_img.shape[1] // 2, :].T, cmap='gray')
            axes[1, 2].imshow(sample_img[:, :, sample_img.shape[2] // 2].T, cmap='gray')
            axes[0, 0].set_title('Input'); axes[1, 0].set_title('Generated')
            plt.tight_layout()

            # Save figure
            sub_fig.savefig(self.results_folder / f'gen_{subject_ids[i]}_comparison.svg')
            wandb.log({f'figures/infer_{subject_ids[i]}': wandb.Image(sub_fig)}, step=self.step)
            plt.close(sub_fig)

            # Save NIfTI
            nifti_img = nib.Nifti1Image(sample_img, affine=val_affine)
            nifti_img.to_filename(str(self.results_folder / f'gen_{subject_ids[i]}_generated.nii.gz'))

