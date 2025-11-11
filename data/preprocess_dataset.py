import os 
import ants 
import json
import traceback
import argparse 
import numpy as np
import nibabel as nib 
import torchio as tio 
from tqdm import tqdm
from skimage import exposure
from collections import defaultdict
import torch 

def adjust_contrast(ants_image, gamma=1.25):
    '''
    Adjust the contrast of an ANTs image using gamma correction.
    Args:
        ants_image (ants.ANTsImage): Input ANTs image.
        gamma (float): Gamma value for adjustment.
    Returns:
        ants.ANTsImage: Contrast-adjusted ANTs image.
    '''
    ants_image_np = ants_image.numpy()
    adjusted_image_np = exposure.adjust_gamma(ants_image_np, gamma=gamma)
    adjusted_image = ants_image.new_image_like(adjusted_image_np)
    return adjusted_image

def normalize_image(image):
    """
    Normalize the image to have zero mean and unit variance.
    Args:
        image: ANTsImage, Nifti1Image, NumPy array, or TorchIO ScalarImage
    Returns:
        Normalized image of the same type as input.
    """
    if isinstance(image, ants.ANTsImage):
        image = ants.iMath(image, 'Normalize')

    elif isinstance(image, nib.Nifti1Image):
        image_np = image.get_fdata()
        image_np = (image_np - np.mean(image_np)) / np.std(image_np)
        image = nib.Nifti1Image(image_np, affine=image.affine) 

    elif isinstance(image, np.ndarray):
        image = (image - np.mean(image)) / np.std(image)

    elif isinstance(image, tio.ScalarImage):
        # image = tio.ZNormalization()(image) mean 0 and std 1
        image = tio.RescaleIntensity((0, 1))(image)

    else:
        raise TypeError("Unsupported image type for normalization.")
    return image

def crop_template(template):
    ''' 
    Crop the template to a target shape.
    Args:
        template (ants.ANTsImage): Input ANTs image template of shape (195, 231, 159).
    Returns:
        ants.ANTsImage: Cropped ANTs image template of shape (192, 192, depth).
    '''
    crop_target_shape = (192, 192, (template.shape[2]//2)*2)
    if isinstance(template, ants.ANTsImage):
        template_np = template.numpy()
        x_start = (template_np.shape[0] - crop_target_shape[0]) // 2
        x_end = x_start + crop_target_shape[0]
        y_start = ((template_np.shape[1] - crop_target_shape[1]) // 2) + 8
        y_end = y_start + crop_target_shape[1]
        z_start = (template_np.shape[2] - crop_target_shape[2]) // 2
        z_end = z_start + crop_target_shape[2]

        lower_ind = [x_start, y_start, z_start]
        upper_ind = [x_end, y_end, z_end] # upper_ind is exclusive in ITK/ANTs, so it's the index AFTER the last voxel

        cropped_template = ants.crop_indices(template, lower_ind, upper_ind)
        return cropped_template

def prepare_template(template_path, adjust=True, gamma=1.75):
    """
    Prepare the template image for registration.
    Args:
        template_path (str): Path to the template NIfTI file.
        adjust (bool): Whether to adjust contrast.
        gamma (float): Gamma value for contrast adjustment.
    Returns:
        ants.ANTsImage: Preprocessed template image.
    """
    template = ants.image_read(template_path)
    template = ants.iMath(template, 'Normalize') 
    if 'T2w' not in template_path:
        template = ants.mask_image(template, ants.get_mask(template))  
    if adjust:
        template = adjust_contrast(template, gamma=gamma)
    template = crop_template(template)
    template = ants.n4_bias_field_correction(template)
    template = ants.iMath(template, 'Normalize')
    template = ants.reorient_image2(template, 'RAS')
    template.set_origin((0, 0, 0))
    return template


def preprocess_nifti(image, template):
    """
    Process a NIfTI image: bias correction, registration, normalization, cropping, and resampling.
    Args:
        image (ants.ANTsImage): Input ANTs image to preprocess.
        template (ants.ANTsImage): Template ANTs image for registration.
    Returns:
        ants.ANTsImage: Preprocessed ANTs image.
    """
    orientation, direction = template.orientation, template.direction
    image.set_origin((0, 0, 0))  
    image = ants.reorient_image2(image, orientation) 
    image.set_direction(direction)

    image = ants.n4_bias_field_correction(image, 1)
    image = ants.iMath(image, 'Normalize')

    image_mask = ants.get_mask(image, low_thresh=0.01, cleanup=1)
    image_masked = ants.mask_image(image, image_mask)

    # Registration
    image_registered = ants.registration(fixed=template, moving=image_masked, type_of_transform='Affine')
    image_bc = image_registered['warpedmovout']
    forward_transform = image_registered['fwdtransforms']

    image_normalized = ants.iMath(image_bc, 'Normalize')

    registration_cost = ants.image_mutual_information(template, image_bc)

    # Crop to 160x160x160
    target_shape = (160, 160, 160)
    tio_image = torch.tensor(image_normalized.numpy()).unsqueeze(0)
    tio_image = tio.ScalarImage(tensor=tio_image, affine=get_affine_from_ants(image_normalized))
    cop = tio.CropOrPad(target_shape, padding_mode='minimum')
    tio_image = cop(tio_image)
    direction = np.array(tio_image.direction).reshape((3, 3)).tolist()
    tio_image = tio.RescaleIntensity((0, 1))(tio_image)
    image_160 = tio_image.data[0, ...].numpy()

    image_final = ants.from_numpy(image_160, spacing=tio_image.spacing, origin=tio_image.origin, direction=direction)
    image_final.set_origin((0, 0, 0))
    return image_final, registration_cost


def save_preprocessed(img, original_path, output_base_dir, nifti_dir):
    """
    Save a preprocessed ANTs image object as a NIfTI file.
    Args:
        img (ants.ANTsImage): Preprocessed ANTs image to save.
        original_path (str): Original file path of the image.
        output_base_dir (str): Base directory to save the preprocessed image.
        nifti_dir (str): Base directory to determine relative paths. Dataset is assumed to be in BIDS format.
        dataset (str): Dataset type, either 'paired' or others.
    Returns:
        None
    """
    relative_path = os.path.relpath(original_path, nifti_dir)
    output_path = os.path.join(output_base_dir, relative_path)
    output_path = output_path.replace('_acq', '_nihpd_acq')
        
    if 'highres' in output_path:
        output_path = output_path.replace('/ses-', '/hf-')

    elif 'lowres' in output_path:
        output_path = output_path.replace('/ses-', '/ulf-')
                    
    # saving the preprocessed nifti
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if isinstance(img, ants.ANTsImage):
        ants.image_write(img, output_path, ri=False)
    elif isinstance(img, nib.Nifti1Image):
        nib.save(img, output_path)
    print(f"Saved {type(img)} preprocessed image to {output_path}", flush=True)

def get_affine_from_ants(img):
    """
    Get the affine transformation matrix from an ANTs image.
    Args:
        img (ants.ANTsImage): Input ANTs image.
    Returns:
        np.ndarray: Affine transformation matrix.
    """
    affine = np.eye(4)
    affine[:3, :3] = img.direction * img.spacing
    affine[:3, 3] = img.origin
    return affine

def get_ants_from_affine(affine):
    """
    Convert a Nifti affine matrix to ANTs image orientation.
    Args:
        affine (np.ndarray): 4x4 Nifti affine matrix.
    Returns:    
        spacing (list): Voxel spacing.
        direction (list): Direction cosines.
        origin (list): Origin coordinates.
    """
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    direction = affine[:3, :3] / spacing[:, np.newaxis]
    origin = affine[:3, 3]

    spacing = spacing.tolist()
    direction = direction.tolist()
    origin = origin.tolist()
    return spacing, direction, origin

def preprocess_all_files(data_dir, output_dir, template_t1, template_t2, transform='Affine'):
    """
    Preprocess T1w and T2w files, create structure for DWI, and handle errors gracefully.
    args:
        data_dir: str, path to directory containing NIfTI files
        output_dir: str, path to output directory
        template_t1: ANTs image object, template for T1w images
        template_t2: ANTs image object, template for T2w images
    returns:
        None
    2 parts: 
    1. Traverse and collect files
    2. Process each T1w, T2w, flair or DWI file
    """ 

    nifti_files = []
    costs_dict = {}
    skipped_files = defaultdict(list)

    for sub in tqdm(sorted(os.listdir(data_dir)), desc="Collecting files"):
        if not sub.startswith('sub-'):
            continue

        try:
            sub_dir = os.path.join(data_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            
            anat_dir = os.path.join(sub_dir, 'anat')
            if os.path.isdir(anat_dir):
                for file in os.listdir(anat_dir):
                    if (file.endswith('.nii.gz') or file.endswith('.nii')) and not file.startswith('._') and ('T1w' in file or 'T2w' in file or 'FLAIR' in file) and not 'heudiconv' in file:
                        nifti_files.append(os.path.join(anat_dir, file))
            
        except Exception as e:
            print(f"Error collecting files for {sub}: {e}", flush=True)
            traceback.print_exc()
            continue

    # save the list of nifti files to a json file
    with open(os.path.join(output_dir, 'nifti_files.json'), 'w') as f:
        for file in nifti_files:
            json.dump(file, f)
            f.write('\n')
    print(f"Collected {len(nifti_files)} NIfTI files for preprocessing.", flush=True)

    ##########################################################
    # 2. Process each T1w, T2w, flair or DWI file
    for file in tqdm(nifti_files, desc="Preprocessing files"):
        try:
            print(f"Processing: {file}", flush=True)
            img = ants.image_read(file)
            if img is None:
                print(f"Skipping file due to load error: {file}", flush=True)
                skipped_files['load_error'].append(file)
                continue

            template = template_t1 if ("T1w" in file or 'dwi' in file) else template_t2 

            # Skip low number of slices
            if img.shape[-1] == 1 or img.shape[-1] == 2 or img.shape[-1] == 3:
                print(f"Skipping file due to small number of slices ({img.shape[-1]}): {file}", flush=True)
                skipped_files['low_num_slices'].append((file, cost))
                continue

            preprocessed_img, cost = preprocess_nifti(img, template=template)
            costs_dict[file] = cost

            if preprocessed_img is None:
                print(f"Skipping file due to preprocessing error: {file}", flush=True)
                skipped_files['preprocessing_error'].append((file, cost))
                continue

            # Skip low cost registrations (larger negative, better)
            if abs(cost) < 0.25: 
                print(f"Cost is too low ({cost}), skipping saving for {file}", flush=True)
                skipped_files['low_cost'].append((file, cost))
                continue

            save_preprocessed(preprocessed_img, file, output_dir, data_dir)

        except Exception as e:
            print(f"Unexpected error with file {file}: {e}", flush=True)
            skipped_files['unexpected_error'].append((file))
            traceback.print_exc()

    # Save metrics and skipped files for review
    metrics = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics, exist_ok=True)

    with open(os.path.join(metrics, f'{transform}_costs.json'), 'w') as f:
        for key, value in costs_dict.items():
            json.dump({key: value}, f)
            f.write('\n')

    # Save skipped files to a JSON file 
    skipped_files_path = os.path.join(metrics, 'skipped_files.json')
    with open(skipped_files_path, 'w') as f:
        json.dump(skipped_files, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NIfTI dataset.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the input NIfTI dataset directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output preprocessed dataset directory.')
    parser.add_argument('--template_t1_path', type=str, required=True, help='Path to the T1w template NIfTI file.')
    parser.add_argument('--template_t2_path', type=str, required=True, help='Path to the T2w template NIfTI file.')
    args = parser.parse_args()

    template_t1 = prepare_template(args.template_t1_path, adjust=True, gamma=1.75)
    template_t2 = prepare_template(args.template_t2_path, adjust=True, gamma=1.25)

    preprocess_all_files(args.data_dir, args.output_dir, template_t1, template_t2, transform='Affine')
    