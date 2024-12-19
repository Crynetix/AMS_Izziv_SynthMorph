#!/usr/bin/env python3

import os
import json
import numpy as np
import nibabel as nib
import tensorflow as tf
import voxelmorph as vxm

############################
# ENV SETUP
############################
# Select GPU if needed, else comment it out
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

############################
# PATH SETUP
############################
DATA_DIR = "/app/data"
OUTPUT_DIR = "/app/output"  # Mounted directory for output displacement fields
JSON_PATH = os.path.join(DATA_DIR, "ThoraxCBCT_dataset.json")

MODEL_DIR = os.path.join(DATA_DIR, "models/2024-12-17/run_03")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "voxelmorph_model_epoch_100.h5")  # adjust if needed

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

############################
# LOAD DATA INFO
############################
with open(JSON_PATH, 'r') as f:
    data_info = json.load(f)

test_pairs = data_info.get("registration_val", [])
if not test_pairs:
    raise ValueError("No test pairs found in the JSON file.")

############################
# UTILITY FUNCTIONS
############################
def load_nifti(path, normalize=True):
    img = nib.load(path)
    vol = img.get_fdata()
    if normalize:
        vol_min, vol_max = np.min(vol), np.max(vol)
        vol = (vol - vol_min) / (vol_max - vol_min + 1e-8)
    vol = vol[..., np.newaxis].astype('float32')
    return vol

def extract_id_from_filename(filepath):
    # filepath example: /app/data/imagesTr/ThoraxCBCT_0011_0001.nii.gz
    # We want to extract 0011_0001 from "ThoraxCBCT_0011_0001.nii.gz"
    base = os.path.basename(filepath)
    # base = ThoraxCBCT_0011_0001.nii.gz
    # Remove prefix "ThoraxCBCT_"
    name = base.replace("ThoraxCBCT_", "")
    # name = 0011_0001.nii.gz
    name = name.replace(".nii.gz", "")
    # name = 0011_0001
    return name

############################
# LOAD MODEL
############################
# Infer input shape from the first pair's fixed image
first_fixed_path = os.path.join(DATA_DIR, test_pairs[0]["fixed"].lstrip("./"))
fixed_vol = load_nifti(first_fixed_path)
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

############################
# RUN INFERENCE AND SAVE FIELDS
############################
for i, pair in enumerate(test_pairs):
    moving_path = os.path.join(DATA_DIR, pair["moving"].lstrip("./"))
    fixed_path = os.path.join(DATA_DIR, pair["fixed"].lstrip("./"))

    # Load volumes
    moving_vol = load_nifti(moving_path, normalize=True)
    fixed_vol = load_nifti(fixed_path, normalize=True)

    # Predict displacement field
    inputs = [moving_vol[np.newaxis, ...], fixed_vol[np.newaxis, ...]]
    pred = vxm_model.predict(inputs)

    # pred[1] contains the displacement field of shape (1, X, Y, Z, 3)
    flow = pred[1][0, ...]

    # Extract IDs for naming
    fixed_id = extract_id_from_filename(fixed_path)
    moving_id = extract_id_from_filename(moving_path)

    # According to naming scheme: disp_<fixed>_<moving>.nii.gz
    disp_filename = f"disp_{fixed_id}_{moving_id}.nii.gz"
    disp_path = os.path.join(OUTPUT_DIR, disp_filename)

    # Save displacement field as NIfTI
    disp_nii = nib.Nifti1Image(flow, np.eye(4))
    nib.save(disp_nii, disp_path)

    print(f"Saved displacement field for pair {i}: {disp_path}")

print("All displacement fields have been saved.")
