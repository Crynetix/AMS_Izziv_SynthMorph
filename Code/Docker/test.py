#!/usr/bin/env python3

import os
import json
import numpy as np
import nibabel as nib
import tensorflow as tf
import voxelmorph as vxm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import neurite as ne  # Ensure you have neurite installed for visualization


# gpuid = 1 -> RTX 2080 Ti
# gpuid = 0 -> RTX 4060 Ti
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Paths inside the container
DATA_DIR = "/app/data"
JSON_PATH = os.path.join(DATA_DIR, "ThoraxCBCT_dataset.json")
MODEL_DIR = os.path.join(DATA_DIR, "models/2024-12-11/run_22")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "voxelmorph_model_epoch_20.h5")  # adjust if needed

RESULTS_DIR = os.path.join(DATA_DIR, "results_no_masks")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Load JSON info
with open(JSON_PATH, 'r') as f:
    data_info = json.load(f)

# Adjust this depending on what your JSON defines for testing
# Here we assume "registration_val" contains pairs of images for validation/testing
test_pairs = data_info.get("registration_val", [])
if not test_pairs:
    raise ValueError("No test pairs found in the JSON file.")

def load_nifti(path, normalize=True):
    """
    Load a NIfTI image and normalize intensities if required.
    
    Args:
        path (str): Path to the NIfTI file.
        normalize (bool): Whether to normalize the intensities.
    
    Returns:
        np.ndarray: The loaded image volume, with a channel dimension added.
    """
    img = nib.load(path)
    vol = img.get_fdata()
    
    if normalize:
        # Min-max normalization
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-8)
    
    vol = vol[..., np.newaxis].astype('float32')
    return vol


# Infer shape from first pair
first_fixed = os.path.join(DATA_DIR, test_pairs[0]["fixed"].lstrip("./"))
fixed_vol = load_nifti(first_fixed)
inshape = fixed_vol.shape[:-1]

nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

vxm_model = vxm.networks.VxmDense(
    inshape=inshape,
    nb_unet_features=nb_features,
    int_steps=0
)

vxm_model.load_weights(MODEL_WEIGHTS)

# Transform model for reference if needed, but we are not warping masks now.
# transform = vxm.networks.Transform(inshape, interp_method='nearest')
"""
for i, pair in enumerate(test_pairs):
    fixed_path = os.path.join(DATA_DIR, pair["fixed"].lstrip("./"))
    moving_path = os.path.join(DATA_DIR, pair["moving"].lstrip("./"))

    fixed_vol = load_nifti(fixed_path)
    moving_vol = load_nifti(moving_path)
    inputs = [moving_vol[np.newaxis,...], fixed_vol[np.newaxis,...]]

    pred = vxm_model.predict(inputs)
    moved_vol = moving_vol + pred  # moved moving image (remove batch dim)
    flow = pred[1][0,...]       # displacement field (X,Y,Z,3)

    # Save the moved image and flow field as NIfTI
    moved_nii = nib.Nifti1Image(moved_vol[...,0], np.eye(4))
    nib.save(moved_nii, os.path.join(RESULTS_DIR, f"moved_image_{i}.nii.gz"))

    flow_nii = nib.Nifti1Image(flow, np.eye(4))
    nib.save(flow_nii, os.path.join(RESULTS_DIR, f"flow_{i}.nii.gz"))

    print(f"Pair {i}:")
    print(f"  Moved image saved to: {os.path.join(RESULTS_DIR, f'moved_image_{i}.nii.gz')}")
    print(f"  Flow field saved to: {os.path.join(RESULTS_DIR, f'flow_{i}.nii.gz')}")

    # Visualization of a middle slice
    z_mid = fixed_vol.shape[2] // 2
    fixed_slice = fixed_vol[...,0][..., z_mid]
    moving_slice = moving_vol[...,0][..., z_mid]
    moved_slice = moved_vol[...,0][..., z_mid]

    flow_slice = flow[:,:,z_mid,:]
    U = flow_slice[...,0]
    V = flow_slice[...,1]
    magnitude = np.sqrt(U**2 + V**2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 0: Fixed, Moving, Moved
    axes[0,0].imshow(fixed_slice, cmap='gray')
    axes[0,0].set_title("Fixed")
    axes[0,0].axis('off')

    axes[0,1].imshow(moving_slice, cmap='gray')
    axes[0,1].set_title("Moving (Original)")
    axes[0,1].axis('off')

    axes[0,2].imshow(moved_slice, cmap='gray')
    axes[0,2].set_title("Moved (Warped Moving)")
    axes[0,2].axis('off')

    # Row 1: Displacement Field and Magnitude
    X = np.arange(flow_slice.shape[1])
    Y = np.arange(flow_slice.shape[0])
    Xgrid, Ygrid = np.meshgrid(X, Y)
    axes[1,0].imshow(fixed_slice, cmap='gray', alpha=0.7)
    axes[1,0].quiver(Xgrid, Ygrid, U, V, color='red', angles='xy', scale_units='xy', scale=1)
    axes[1,0].set_title("Displacement Field (Quiver)")
    axes[1,0].axis('off')

    axes[1,1].imshow(magnitude, cmap='jet')
    axes[1,1].set_title("Displacement Magnitude")
    axes[1,1].axis('off')

    # The last subplot can just remain blank or show something else if desired
    axes[1,2].axis('off')
    axes[1,2].set_title("No Masks/No Segments")

    fig.suptitle(f"Pair {i}", fontsize=16)
    plt.tight_layout()

    png_path = os.path.join(RESULTS_DIR, f"preview_{i}.png")
    plt.savefig(png_path)
    plt.close(fig)

    print(f"  Preview slice saved to: {png_path}")
"""

for i, pair in enumerate(test_pairs):
    moving_path = os.path.join(DATA_DIR, pair["moving"].lstrip("./"))
    fixed_path = os.path.join(DATA_DIR, pair["fixed"].lstrip("./"))

    # Load and normalize images
    moving_vol = load_nifti(moving_path, normalize=True)
    fixed_vol = load_nifti(fixed_path, normalize=True)

    # Prepare inputs and run prediction
    inputs = [moving_vol[np.newaxis, ...], fixed_vol[np.newaxis, ...]]
    pred = vxm_model.predict(inputs)

    moved_vol = pred[0][0, ..., 0]  # Remove batch and channel dimensions
    flow = pred[1][0, ...]          # Displacement field

    # Save results as NIfTI
    nib.save(nib.Nifti1Image(moved_vol, np.eye(4)), os.path.join(RESULTS_DIR, f"moved_image_{i}.nii.gz"))
    nib.save(nib.Nifti1Image(flow, np.eye(4)), os.path.join(RESULTS_DIR, f"flow_{i}.nii.gz"))

    # Visualization
    fixed_slice = fixed_vol[..., 0][:, :, fixed_vol.shape[2] // 2]
    moving_slice = moving_vol[..., 0][:, :, moving_vol.shape[2] // 2]
    moved_slice = moved_vol[:, :, fixed_vol.shape[2] // 2]

    # Flow visualization (magnitude)
    flow_slice = flow[:, :, fixed_vol.shape[2] // 2, :]
    flow_magnitude = np.sqrt(np.sum(np.square(flow_slice), axis=-1))

    # Visualization using Neurite
    images = [moving_slice, fixed_slice, moved_slice, flow_magnitude]
    titles = ['Moving', 'Fixed', 'Moved', 'Flow Magnitude']
    ne.plot.slices(images, titles=titles, cmaps=['gray', 'gray', 'gray', 'jet'], do_colorbars=True)

    # Save the visualization as a PNG
    plt.savefig(os.path.join(RESULTS_DIR, f"preview_{i}.png"))
    plt.close()

    print(f"Visualization saved for pair {i}: {os.path.join(RESULTS_DIR, f'preview_{i}.png')}")

transform = vxm.networks.Transform(inshape, interp_method='nearest')

def compute_dice(vol1, vol2):
    intersection = np.sum((vol1 > 0) & (vol2 > 0))
    size1 = np.sum(vol1 > 0)
    size2 = np.sum(vol2 > 0)
    if size1 + size2 == 0:
        return 1.0  # if both are empty, Dice = 1
    dice = 2.0 * intersection / (size1 + size2)
    return dice

dices = []
for i, pair in enumerate(test_pairs):
    fixed_path = os.path.join(DATA_DIR, pair["fixed"].lstrip("./"))
    moving_path = os.path.join(DATA_DIR, pair["moving"].lstrip("./"))

    # Derive mask paths (assuming same filename pattern)
    fixed_mask_path = fixed_path.replace("imagesTs", "masksTs")
    moving_mask_path = moving_path.replace("imagesTs", "masksTs")

    fixed_vol = load_nifti(fixed_path, normalize = True)
    moving_vol = load_nifti(moving_path, normalize = True)
    inputs = [moving_vol[np.newaxis,...], fixed_vol[np.newaxis,...]]

    pred = vxm_model.predict(inputs)
    moved_vol = pred[0][0,...]  # moved moving image (remove batch dim)
    flow = pred[1][0,...]       # displacement field (X,Y,Z,3)

    # Load masks
    fixed_mask = load_nifti(fixed_mask_path)[...,0]  # shape (X,Y,Z)
    moving_mask = load_nifti(moving_mask_path)[...,0]

    # Warp moving mask
    warped_mask = transform.predict([moving_mask[np.newaxis,...,np.newaxis], pred[1]])[0,...,0]

    # Compute dice between fixed_mask and warped_mask
    warped_mask_bin = np.round(warped_mask).astype(np.int32)
    dice_val = compute_dice(fixed_mask, warped_mask_bin)
    dices.append(dice_val)

    # Save results as NIfTI
    moved_nii = nib.Nifti1Image(moved_vol[...,0], np.eye(4))
    nib.save(moved_nii, os.path.join(RESULTS_DIR, f"moved_image_{i}.nii.gz"))

    warped_mask_nii = nib.Nifti1Image(warped_mask_bin.astype(np.float32), np.eye(4))
    nib.save(warped_mask_nii, os.path.join(RESULTS_DIR, f"warped_mask_{i}.nii.gz"))

    flow_nii = nib.Nifti1Image(flow, np.eye(4))
    nib.save(flow_nii, os.path.join(RESULTS_DIR, f"flow_{i}.nii.gz"))

    print(f"Pair {i}:")
    print(f"  Dice score: {dice_val:.4f}")
    print(f"  Moved image saved to: {os.path.join(RESULTS_DIR, f'moved_image_{i}.nii.gz')}")
    print(f"  Warped mask saved to: {os.path.join(RESULTS_DIR, f'warped_mask_{i}.nii.gz')}")
    print(f"  Flow field saved to: {os.path.join(RESULTS_DIR, f'flow_{i}.nii.gz')}")

    # Optional visualization of a middle slice:
    z_mid = fixed_vol.shape[2] // 2
    fixed_slice = fixed_vol[...,0][..., z_mid]
    moving_slice = moving_vol[...,0][..., z_mid]
    moved_slice = moved_vol[...,0][..., z_mid]
    warped_mask_slice = warped_mask_bin[..., z_mid]

    # Extract the flow slice and prepare displacement field plots
    flow_slice = flow[:,:,z_mid,:]  # shape (X,Y,3)
    U = flow_slice[...,0]  # displacement in X
    V = flow_slice[...,1]  # displacement in Y
    magnitude = np.sqrt(U**2 + V**2)

    # Extract masks for the slice
    fixed_mask_slice = fixed_mask[..., z_mid]
    moving_mask_slice = moving_mask[..., z_mid]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Row 0: Fixed, Moving (original), Moved
    axes[0,0].imshow(fixed_slice, cmap='gray')
    axes[0,0].set_title("Fixed")
    axes[0,0].axis('off')

    axes[0,1].imshow(moving_slice, cmap='gray')
    axes[0,1].set_title("Moving (Original)")
    axes[0,1].axis('off')

    axes[0,2].imshow(moved_slice, cmap='gray')
    axes[0,2].set_title("Moved (Warped Moving)")
    axes[0,2].axis('off')

    # Row 1: Fixed + Warped Mask, Displacement field, Displacement magnitude
    overlay = np.dstack([fixed_slice, fixed_slice, fixed_slice])
    overlay[warped_mask_slice > 0, :] = [1, 0, 0]  # red overlay where mask > 0
    axes[1,0].imshow(overlay)
    axes[1,0].set_title("Fixed + Warped Mask")
    axes[1,0].axis('off')

    X = np.arange(flow_slice.shape[1])
    Y = np.arange(flow_slice.shape[0])
    Xgrid, Ygrid = np.meshgrid(X, Y)
    axes[1,1].imshow(fixed_slice, cmap='gray', alpha=0.7)  # background
    axes[1,1].quiver(Xgrid, Ygrid, U, V, color='red', angles='xy', scale_units='xy', scale=1)
    axes[1,1].set_title("Displacement Field (Quiver)")
    axes[1,1].axis('off')

    axes[1,2].imshow(magnitude, cmap='jet')
    axes[1,2].set_title("Displacement Magnitude")
    axes[1,2].axis('off')

    # Row 2: Fixed Mask alone, Moving Mask alone, Warped Mask alone
    axes[2,0].imshow(fixed_mask_slice, cmap='gray')
    axes[2,0].set_title("Fixed Mask")
    axes[2,0].axis('off')

    axes[2,1].imshow(moving_mask_slice, cmap='gray')
    axes[2,1].set_title("Moving Mask (Original)")
    axes[2,1].axis('off')

    axes[2,2].imshow(warped_mask_slice, cmap='gray')
    axes[2,2].set_title("Warped Mask")
    axes[2,2].axis('off')

    fig.suptitle(f"Pair {i} - Dice: {dice_val:.4f}", fontsize=16)
    plt.tight_layout()

    png_path = os.path.join(RESULTS_DIR, f"preview_{i}.png")
    plt.savefig(png_path)
    plt.close(fig)

    print(f"  Preview slice saved to: {png_path}")


mean_dice = np.mean(dices) if dices else 0.0
print(f"\nMean Dice on test set: {mean_dice:.4f}")
