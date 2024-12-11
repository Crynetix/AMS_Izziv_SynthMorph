#!/usr/bin/env python3

import os
import json
import numpy as np
import nibabel as nib
import tensorflow as tf
import voxelmorph as vxm

# -----------------------
# Hardcoded Parameters
# -----------------------
DATA_DIR = "/app/data"  # directory containing the JSON and data subfolders
JSON_PATH = os.path.join(DATA_DIR, "ThoraxCBCT_dataset.json")
EPOCHS = 50
STEPS_PER_EPOCH = 10
BATCH_SIZE = 4
MODEL_DIR = os.path.join(DATA_DIR, "models")
GPU_ID = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

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
def load_nifti(path):
    img = nib.load(path)
    vol = img.get_fdata()  # shape: (X, Y, Z)
    # Add a channel dimension at the end (X, Y, Z, 1)
    vol = vol[..., np.newaxis]
    return vol.astype('float32')

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
            moving_vol = load_nifti(m_path)
            fixed_vol = load_nifti(f_path)
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

losses = [
    vxm.losses.NCC().loss,     # similarity loss
    vxm.losses.Grad('l2').loss # smoothness loss
]
loss_weights = [1.0, 0.01]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'voxelmorph_model_{epoch:02d}.h5'),
    save_freq='epoch',
    save_weights_only=True
)

# -----------------------
# Train the Model
# -----------------------
vxm_model.fit(
    train_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback]
)
