#!/usr/bin/env python3

import os
import json
import numpy as np
import nibabel as nib
import tensorflow as tf
import voxelmorph as vxm
#import wandb
from datetime import datetime
from scipy.ndimage import zoom
#from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


# -----------------------
# Hardcoded Parameters
# -----------------------
DATA_DIR = "/app/data"  # directory containing the JSON and data subfolders
JSON_PATH = os.path.join(DATA_DIR, "ThoraxCBCT_dataset.json")
EPOCHS = 20
STEPS_PER_EPOCH = 100
BATCH_SIZE = 1
MODEL_DIR = os.path.join(DATA_DIR, "models")
LEARNING_RATE = 0.0001 
GPU_ID = '0' # 0 -> RTX 2080 Ti; 1 -> RTX 4060 Ti


# Create model directory with date and unique run number
current_date = datetime.now().strftime("%Y-%m-%d")
date_dir = os.path.join(MODEL_DIR, current_date)

# Ensure the date directory exists
if not os.path.exists(date_dir):
    os.makedirs(date_dir)

# Find the next run number
existing_runs = [d for d in os.listdir(date_dir) if os.path.isdir(os.path.join(date_dir, d))]
next_run_number = len(existing_runs) + 1
run_dir = os.path.join(date_dir, f"run_{next_run_number:02d}")

# Create the run directory
os.makedirs(run_dir)



os.environ['NVIDIA_VISIBLE_DEVICES'] = GPU_ID
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#wandb.login(key = "73b7638455edcf4aa24699fa3ee6d3445074b620")
# -----------------------
# Load JSON and Extract Training Pairs
# -----------------------
with open(JSON_PATH, 'r') as f:
    data_info = json.load(f)

training_pairs = data_info.get("training_paired_images", [])
# Filter out pairs with ThoraxCBCT_0000 to exclude broken images

filtered_pairs = []
for pair in training_pairs:
    fixed_path = pair["fixed"]
    moving_path = pair["moving"]
    filtered_pairs.append((moving_path, fixed_path))


if len(filtered_pairs) == 0:
    raise ValueError("No valid training pairs found after filtering.")

# Convert relative paths to absolute paths
def abs_path(rel_path):
    return os.path.join(DATA_DIR, rel_path.lstrip("./"))

filtered_pairs = [(abs_path(m), abs_path(f)) for (m, f) in filtered_pairs]

# -----------------------
# Data Generator
# -----------------------
def load_nifti(path, normalize=True, downsample_factor=0.5):
    """
    Load a NIfTI image, normalize intensities if required, and downsample.

    Args:
        path (str): Path to the NIfTI file.
        normalize (bool): Whether to normalize the intensities.
        downsample_factor (float): Factor by which to downsample the image in all dimensions.

    Returns:
        np.ndarray: The loaded image volume, with a channel dimension added.
    """
    img = nib.load(path)
    vol = img.get_fdata()

    # Downsample the image using scipy's zoom
    if downsample_factor and downsample_factor != 1.0:
        vol = zoom(vol, zoom=downsample_factor, order=1)  # Linear interpolation

    if normalize:
        # Min-max normalization
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-8)

    vol = vol[..., np.newaxis].astype('float32')
    return vol

def data_generator(pairs, batch_size):
    # pairs is a list of (moving_path, fixed_path)
    # On-the-fly loading for each batch
    while True:
        # randomly select batch_size pairs
        idxs = np.random.randint(0, len(pairs), batch_size)
        moving_batch = []
        fixed_batch = []
        for i in idxs:
            m_path, f_path = pairs[i]
            moving_vol = load_nifti(m_path, normalize = True, downsample_factor = 0.5)
            fixed_vol = load_nifti(f_path, normalize = True, downsample_factor = 0.5)
            moving_batch.append(moving_vol)
            fixed_batch.append(fixed_vol)
        moving_batch = np.stack(moving_batch, axis=0)
        fixed_batch = np.stack(fixed_batch, axis=0)

        # inputs: [moving, fixed]
        inputs = [moving_batch, fixed_batch]

        # outputs: [fixed_image, zero_phi]
        # zero_phi is a zero array the size of the deformation field, 
        # deformation field has shape (X, Y, Z, 3) for 3D
        shape = fixed_batch.shape[1:-1]  # e.g., (256, 192, 192)
        zero_phi = np.zeros((batch_size, *shape, 3), dtype='float32')

        outputs = [fixed_batch, zero_phi]
        yield (inputs, outputs)

train_gen = data_generator(filtered_pairs, BATCH_SIZE)

# -----------------------
# Build Model
# -----------------------
# We assume all volumes have the same shape. Let's infer from the first pair:
test_vol = load_nifti(filtered_pairs[0][0])  # load one volume to get shape
inshape = test_vol.shape[:-1]  # (X, Y, Z)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

vxm_model = vxm.networks.VxmDense(
    inshape=inshape,
    nb_unet_features=nb_features,
    int_steps=0  # no diffeomorphic integration steps for now
)

# Define loss functions
losses = [
    vxm.losses.NCC().loss,                 # Similarity loss
    vxm.losses.Grad('l2').loss,           # Smoothness loss
    tf.keras.losses.MeanSquaredError(),   # MSE loss (optional additional loss)
]

loss_weights = [1.0, 0.01, 0.1]  # Adjust weights for each loss term

vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=losses, loss_weights=loss_weights)

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

#wandb.init(
#    project="voxelmorph_training",  # Replace with your project name
#    name="voxelmorph_experiment",  # Optional: unique run name
#    config={
#        "epochs": EPOCHS,
#        "batch_size": BATCH_SIZE,
#        "learning_rate": LEARNING_RATE,
#        "nb_features": nb_features,
#        "int_steps": 0
#    }
#)

# Updated ModelCheckpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(run_dir, 'voxelmorph_model_epoch_{epoch:02d}.h5'),
    save_freq='epoch',
    save_weights_only=True
)

# Add WandB callback to track training metrics
#wandb_callback = WandbEvalCallback(
#    save_model=False,  # Prevent uploading model weights unless desired
#    log_weights=False,  # Log model weights (optional, for debugging)
#    log_gradients=False  # Log gradient norms (optional, for debugging)
#)

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.5, patience=5, min_lr=1e-6
)

# -----------------------
# Train the Model
# -----------------------
vxm_model.fit(
    train_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, lr_callback]
)

print(f"Model weights will be saved to: {run_dir}")
