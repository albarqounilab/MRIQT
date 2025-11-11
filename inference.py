#-*- coding:utf-8 -*-
# +
import torch
import numpy as np
import argparse
from torchvision.transforms import Compose, Lambda

import ants 
import nibabel as nib

from diffusion_model.unet import create_model
from diffusion_model.trainer import MRIQT
from data.preprocess_dataset import preprocess_nifti, prepare_template

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class Inferrer:
    def __init__(self, model, checkpoint_path, device='cuda'):
        """
        Inferrer class for loading a trained diffusion model and running inference.
        Args:
            model: Diffusion model instance.
            checkpoint_path: Path to the model checkpoint.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        """
        Load model weights from checkpoint.
        Args:
            checkpoint_path: Path to the model checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['ema'], strict=False)
        print(f"Loaded model from {checkpoint_path} at step {checkpoint['step']}.") 

    @torch.inference_mode()
    def infer(self, condition_tensors, strength=0.65, sampler="ddpm", clip_denoised=True, results_folder='./results'):
        """Run image-to-image translation using the loaded model.
        Args: 
            condition_tensors: low-quality data, shape (B, C, D, H, W)
            strength: noise strength, between 0 and 1
            sampler: "ddpm" or "ddim"
            clip_denoised: whether to clip denoised outputs
            results_folder: folder to save results
        Returns:
            generated high-quality data, shape (B, C, D, H, W)
        """
        with torch.inference_mode():
            generated = self.model.img2img(
                condition_tensors=condition_tensors.to(self.device),
                strength=strength,
                sampler=sampler,
                clip_denoised=clip_denoised,
                results_folder=results_folder
            )
        return generated

@torch.inference_mode()
def main(args):
    # ============================================================
    # 1. Load ultra-low-field image and template then preprocess
    # ============================================================
    template = ants.image_read('data_preprocessing/nihpd_asym_00-02_t1w.nii')
    template = prepare_template(template)
    
    ulf = ants.image_read(args.ulf_path)
    preprocessed_ulf = preprocess_nifti(ulf, template)

    transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: 2 * (t - t.min()) / (t.max() - t.min() + 1e-8) - 1), 
        Lambda(lambda t: t.unsqueeze(0)),  
        Lambda(lambda t: t.permute(0, 3, 2, 1)),  
    ])
    input_ulf = transform(preprocessed_ulf).unsqueeze(0).cuda()  # add batch dimension
    # ============================================================
    # 2. Load MRIQT and inferrer
    # ============================================================
    unet = create_model(
            image_size=160,
            num_channels=64,
            num_res_blocks=2,
            attention_resolutions="20,10",
            in_channels=2,
            out_channels=1,
        ).cuda()

    mriqt = MRIQT(
        unet, 
        image_size=160,
        depth_size=160,
        perceptual_loss_fn=None,
        cond_drop_prob=0.1,
        guidance_weight=2.,
        use_cfg=True, 
    ).cuda()
    mriqt.eval()
    
    inferrer = Inferrer(
        model = mriqt,
        checkpoint_path=args.checkpoint,
        device='cuda',
    )
    # ============================================================
    # 3. Run inference and preprocess output
    # ============================================================
    generated = inferrer.infer(input_ulf)
    generated_np = generated.detach().cpu().numpy()
    generated_np = generated_np.permute(0, 1, 4, 3, 2)[0,0]
    generated_np = (generated_np - generated_np.min()) / (generated_np.max() - generated_np.min() + 1e-8)
    generated_nii = nib.Nifti1Image(generated_np, affine=np.eye(4))
    nib.save(generated_nii, 'generated_super_field.nii.gz')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MRIQT Inference Script')
    parser.add_argument('--ulf_path', type=str, required=True,
                        help='Path to the ultra-low-field NIfTI image.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained MRIQT model checkpoint.')
    
    args = parser.parse_args()
    main(args)