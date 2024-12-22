#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
import voxelmorph as vxm

def parse_args():
    parser = argparse.ArgumentParser(description="Run model inference and output deformation fields.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained model weights (e.g., /app/data/models/.../voxelmorph_model.h5)."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="registration_val",
        help="Key in the JSON file indicating which pairs to use (e.g., registration_val, registration_test)."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="/app/data/ThoraxCBCT_dataset.json",
        help="Path to the JSON file that contains the test pairs."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/app/output",
        help="Directory to save the displacement fields (nii.gz)."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Select which GPU to use (e.g., 0 or 1). Set to '' to disable GPU."
    )
    return parser.parse_args()


def load_nifti(path, normalize=True):
    img = nib.load(path)
    vol = img.get_fdata()
    if normalize:
        vol_min, vol_max = np.min(vol), np.max(vol)
        vol = (vol - vol_min) / (vol_max - vol_min + 1e-8)
    vol = vol[..., np.newaxis].astype('float32')
    return vol

def extract_id_from_filename(filepath):
    # Example: "/app/data/imagesTr/ThoraxCBCT_0011_0001.nii.gz" -> "0011_0001"
    base = os.path.basename(filepath)
    name = base.replace("ThoraxCBCT_", "").replace(".nii.gz", "")
    return name


def main():
    args = parse_args()
    
    # Setup environment
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        # If empty, means we don't use any GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load the JSON file
    with open(args.json, 'r') as f:
        data_info = json.load(f)

    test_pairs = data_info.get(args.subset, [])
    if not test_pairs:
        raise ValueError(f"No pairs found under the key '{args.subset}' in the JSON file.")

    # Infer shape from the first fixed image
    first_fixed_path = os.path.join(os.path.dirname(args.json), test_pairs[0]["fixed"].lstrip("./"))
    fixed_vol = load_nifti(first_fixed_path)
    inshape = fixed_vol.shape[:-1]

    # Build model
    nb_features = [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16]
    ]
    vxm_model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=nb_features,
        int_steps=0
    )

    print(f"Loading weights from: {args.weights}")
    vxm_model.load_weights(args.weights)

    # For each pair, run inference and save displacement field
    for i, pair in enumerate(test_pairs):
        moving_path = os.path.join(os.path.dirname(args.json), pair["moving"].lstrip("./"))
        fixed_path = os.path.join(os.path.dirname(args.json), pair["fixed"].lstrip("./"))

        # Load volumes
        moving_vol = load_nifti(moving_path)
        fixed_vol = load_nifti(fixed_path)

        # Predict
        inputs = [moving_vol[np.newaxis, ...], fixed_vol[np.newaxis, ...]]
        pred = vxm_model.predict(inputs)

        flow = pred[1][0, ...]  # shape (X, Y, Z, 3)

        # Construct output filename
        fixed_id = extract_id_from_filename(fixed_path)
        moving_id = extract_id_from_filename(moving_path)
        disp_filename = f"disp_{fixed_id}_{moving_id}.nii.gz"
        disp_path = os.path.join(args.output, disp_filename)

        nib.save(nib.Nifti1Image(flow, np.eye(4)), disp_path)
        print(f"Saved deformation field: {disp_path}")

    print("All deformation fields have been saved.")


if __name__ == "__main__":
    main()
