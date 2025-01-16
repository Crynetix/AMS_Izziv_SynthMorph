#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
import voxelmorph as vxm
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from datetime import datetime
from scipy.ndimage import zoom


def parse_args():
    """
    Parse command-line arguments for training VoxelMorph.
    """
    parser = argparse.ArgumentParser(description="Train VoxelMorph model.")

    # Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/app/data",
        help="Directory containing data and JSON (default: /app/data)",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="ThoraxCBCT_dataset.json",
        help="Name of the JSON file relative to data_dir (default: ThoraxCBCT_dataset.json)",
    )
    parser.add_argument(
        "--train_key",
        type=str,
        default="training_paired_images",
        help="Key in the JSON for training pairs, for this model please use only image pairs as keypoints arent implemented (default: training_paired_images)",
    )
    parser.add_argument(
        "--val_key",
        type=str,
        default="registration_val",
        help="Key in the JSON for validation pairs (default: registration_val)",
    )

    # GPU
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU ID to use. Example: 0 or 1. (default: 0)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=40,
        help="Steps per epoch (default: 40)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )

    # Data preprocessing
    parser.add_argument(
        "--downsample_factor",
        type=float,
        default=1.0,
        help="Downsample factor in all dims, if downsampling is needed 0.5 is recommended (default: 1.0)",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_false",
        dest="normalize",   # the actual variable name becomes "normalize"
        default=True,
        help="Call to disable min-max normalization (default: Enabled)."
    )


    # Weights & Biases
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="voxelmorph_training",
        help="W&B Project name (default: voxelmorph_training)",
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default="",
        help="W&B API key (if needed to login). By default empty if already logged in.",
    )

    return parser.parse_args()


def load_nifti(path, normalize=False, downsample_factor=1.0):
    """
    Load a NIfTI image, optionally normalize intensities, and downsample.
    """
    import nibabel as nib
    from scipy.ndimage import zoom

    img = nib.load(path)
    vol = img.get_fdata()

    # Downsample
    if downsample_factor != 1.0:
        vol = zoom(vol, zoom=downsample_factor, order=1)  # linear interpolation

    if normalize:
        vol_min, vol_max = np.min(vol), np.max(vol)
        vol = (vol - vol_min) / (vol_max - vol_min + 1e-8)

    vol = vol[..., np.newaxis].astype('float32')
    return vol


def data_generator(pairs, batch_size, normalize=False, downsample_factor=1.0):
    """
    Yields batches for Voxelmorph training: [moving, fixed] -> [fixed, zero_phi].
    """
    import numpy as np

    while True:
        idxs = np.random.randint(0, len(pairs), batch_size)
        moving_batch = []
        fixed_batch = []
        for i in idxs:
            m_path, f_path = pairs[i]
            moving_vol = load_nifti(m_path, normalize=normalize, downsample_factor=downsample_factor)
            fixed_vol = load_nifti(f_path, normalize=normalize, downsample_factor=downsample_factor)
            moving_batch.append(moving_vol)
            fixed_batch.append(fixed_vol)

        moving_batch = np.stack(moving_batch, axis=0)
        fixed_batch = np.stack(fixed_batch, axis=0)

        # inputs: [moving, fixed]
        inputs = [moving_batch, fixed_batch]

        # outputs: [fixed_batch, zero_phi]
        shape = fixed_batch.shape[1:-1]  # e.g. (256,192,192)
        zero_phi = np.zeros((batch_size, *shape, 3), dtype='float32')
        outputs = [fixed_batch, zero_phi]

        yield (inputs, outputs)


def main():
    args = parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # Setup directories
    DATA_DIR = args.data_dir
    JSON_PATH = os.path.join(DATA_DIR, args.json_path)
    MODEL_DIR = os.path.join(DATA_DIR, "models")

    # W&B Login if key is provided
    if args.wandb_key.strip():
        import wandb
        wandb.login(key=args.wandb_key.strip())

    # Initialize W&B Run
    wandb.init(
        project=args.wandb_project,
        config={
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "downsample_factor": args.downsample_factor,
            "normalize": args.normalize,
        }
    )
    config = wandb.config  # shorthand

    # Load JSON
    with open(JSON_PATH, 'r') as f:
        data_info = json.load(f)

    # Get training pairs
    train_key = args.train_key
    val_key = args.val_key
    training_pairs_json = data_info.get(train_key, [])
    if not training_pairs_json:
        raise ValueError(f"No training pairs found under key '{train_key}' in {JSON_PATH}")

    # Build training pairs as absolute paths
    def abs_path(rel_path):
        return os.path.join(DATA_DIR, rel_path.lstrip("./"))

    training_pairs = []
    for p in training_pairs_json:
        moving_path = abs_path(p["moving"])
        fixed_path = abs_path(p["fixed"])
        training_pairs.append((moving_path, fixed_path))

    # Get validation pairs
    validation_pairs_json = data_info.get(val_key, [])
    validation_pairs = []
    if validation_pairs_json:
        for p in validation_pairs_json:
            moving_path = abs_path(p["moving"])
            fixed_path = abs_path(p["fixed"])
            validation_pairs.append((moving_path, fixed_path))
    else:
        # fallback: manual split if no val data in JSON
        split_idx = int(0.8 * len(training_pairs))
        validation_pairs = training_pairs[split_idx:]
        training_pairs = training_pairs[:split_idx]

    print(f"Number of training pairs: {len(training_pairs)}")
    print(f"Number of validation pairs: {len(validation_pairs)}")

    # Create generators
    train_gen = data_generator(
        training_pairs,
        config.batch_size,
        normalize=config.normalize,
        downsample_factor=config.downsample_factor
    )
    val_gen = data_generator(
        validation_pairs,
        config.batch_size,
        normalize=config.normalize,
        downsample_factor=config.downsample_factor
    )

    # Infer shape from the first training pair
    test_vol = load_nifti(training_pairs[0][0], normalize=config.normalize, downsample_factor=config.downsample_factor)
    inshape = test_vol.shape[:-1]
    print(f"Training image shape: {inshape}")

    # Build the VoxelMorph model
    nb_features = [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16]
    ]

    vxm_model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=nb_features,
        int_steps=0
    )

    # Define losses and optimizer
    losses = [
        vxm.losses.NCC().loss,
        vxm.losses.Grad('l2').loss,
    ]
    loss_weights = [1.0, 0.01]

    vxm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=losses,
        loss_weights=loss_weights
    )

    # Directory for model checkpoints
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(MODEL_DIR, current_date)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)

    existing_runs = [d for d in os.listdir(date_dir) if os.path.isdir(os.path.join(date_dir, d))]
    next_run_number = len(existing_runs) + 1
    run_dir = os.path.join(date_dir, f"run_{next_run_number:02d}")
    os.makedirs(run_dir)
    checkpoint_path = os.path.join(run_dir, 'voxelmorph_model_epoch_{epoch:02d}.h5')

    # W&B Callbacks
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

    metrics_logger = WandbMetricsLogger()

    model_ckpt = WandbModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=True
    )

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=5, min_lr=1e-6
    )

    # TRAIN
    vxm_model.fit(
        train_gen,
        steps_per_epoch=config.steps_per_epoch,
        epochs=config.epochs,
        validation_data=val_gen,
        validation_steps=len(validation_pairs) // config.batch_size if validation_pairs else None,
        callbacks=[metrics_logger, model_ckpt, lr_callback]
    )

    print(f"Model weights have been saved to: {run_dir}")


if __name__ == "__main__":
    main()
