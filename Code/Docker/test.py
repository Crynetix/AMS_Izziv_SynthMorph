import os
import json
import numpy as np
import nibabel as nib
import tensorflow as tf
import voxelmorph as vxm

# Paths inside the container
DATA_DIR = "/app/data"
JSON_PATH = os.path.join(DATA_DIR, "ThoraxCBCT_dataset.json")
MODEL_DIR = os.path.join(DATA_DIR, "models")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "voxelmorph_model_50.h5")  # adjust if needed

# Load JSON info
with open(JSON_PATH, 'r') as f:
    data_info = json.load(f)

test_pairs = data_info.get("test_paired_images", [])
if not test_pairs:
    raise ValueError("No test pairs found in the JSON file.")

# Load model (same architecture as training)
# Determine volume shape by loading one test image
def load_nifti(path):
    img = nib.load(path)
    vol = img.get_fdata()
    vol = vol[..., np.newaxis].astype('float32')
    return vol

# We'll just load the first pair to infer shape
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

# We need a transform model to warp masks
transform = vxm.networks.Transform(inshape, interp_method='nearest')

def compute_dice(vol1, vol2):
    # vol1 and vol2 are binary masks (0 or 1)
    intersection = np.sum((vol1 > 0) & (vol2 > 0))
    size1 = np.sum(vol1 > 0)
    size2 = np.sum(vol2 > 0)
    if size1 + size2 == 0:
        return 1.0  # if both are empty, Dice = 1
    dice = 2.0 * intersection / (size1 + size2)
    return dice

dices = []
for pair in test_pairs:
    fixed_path = os.path.join(DATA_DIR, pair["fixed"].lstrip("./"))
    moving_path = os.path.join(DATA_DIR, pair["moving"].lstrip("./"))

    # Derive mask paths (assuming same filename pattern)
    fixed_mask_path = fixed_path.replace("imagesTs", "masksTs")
    moving_mask_path = moving_path.replace("imagesTs", "masksTs")

    fixed_vol = load_nifti(fixed_path)
    moving_vol = load_nifti(moving_path)
    inputs = [moving_vol[np.newaxis,...], fixed_vol[np.newaxis,...]]

    pred = vxm_model.predict(inputs)
    moved_vol = pred[0]  # moved moving image
    flow = pred[1]       # displacement field

    # Load masks
    fixed_mask = load_nifti(fixed_mask_path)[...,0]  # shape (X,Y,Z)
    moving_mask = load_nifti(moving_mask_path)[...,0]

    # Warp moving mask
    warped_mask = transform.predict([moving_mask[np.newaxis,...,np.newaxis], flow])[0,...,0]

    # Compute dice between fixed_mask and warped_mask
    # Round warped_mask to binary since nearest should have kept it binary
    warped_mask_bin = np.round(warped_mask).astype(np.int32)
    dice_val = compute_dice(fixed_mask, warped_mask_bin)
    dices.append(dice_val)

mean_dice = np.mean(dices)
print(f"Mean Dice on test set: {mean_dice:.4f}")
