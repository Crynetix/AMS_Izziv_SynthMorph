#!/usr/bin/env python3

import os
import json
import numpy as np
import nibabel as nib
import tensorflow as tf
import voxelmorph as vxm
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback
from datetime import datetime
from scipy.ndimage import zoom

#######################
# CONFIG AND ENV SETUP #
#######################
DATA_DIR = "/app/data"  # directory containing the JSON and data subfolders
JSON_PATH = os.path.join(DATA_DIR, "ThoraxCBCT_dataset.json")
MODEL_DIR = os.path.join(DATA_DIR, "models")
# gpuid = 1 -> RTX 2080 Ti
# gpuid = 0 -> RTX 4060 Ti
GPU_ID = '0'  # GPU selection if multiple GPUs available

os.environ['NVIDIA_VISIBLE_DEVICES'] = GPU_ID
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

##################
# WANDB INIT     #
##################
wandb.login(key = "73b7638455edcf4aa24699fa3ee6d3445074b620")

# Initialize a W&B run with some default configs. You can override these from CLI or W&B UI.
wandb.init(project="voxelmorph_training", config={
    "epochs": 250,
    "steps_per_epoch": 40,
    "batch_size": 1,
    "learning_rate": 0.0001,
    "downsample_factor": 0.5,
    "normalize": True
})
config = wandb.config  # shorthand

##########################
# LOAD AND SPLIT DATA    #
##########################
with open(JSON_PATH, 'r') as f:
    data_info = json.load(f)

training_pairs = data_info.get("training_paired_images", [])
if len(training_pairs) == 0:
    raise ValueError("No training pairs found in JSON file.")

# If your JSON has a validation list, e.g., 'registration_val' or 'validate_paired_images'
# Adjust this line according to your actual JSON structure
validation_pairs = data_info.get("registration_val", [])

def abs_path(rel_path):
    return os.path.join(DATA_DIR, rel_path.lstrip("./"))

training_pairs = [(abs_path(m), abs_path(f)) for (m, f) in [(p["moving"], p["fixed"]) for p in training_pairs]]
if validation_pairs:
    validation_pairs = [(abs_path(p["moving"]), abs_path(p["fixed"])) for p in validation_pairs]
else:
    # If no validation pairs are provided, you can split training data manually
    # e.g., 80/20 split
    split_idx = int(0.8 * len(training_pairs))
    validation_pairs = training_pairs[split_idx:]
    training_pairs = training_pairs[:split_idx]

print(f"Number of training pairs: {len(training_pairs)}")
print(f"Number of validation pairs: {len(validation_pairs)}")

######################
# DATA GENERATOR     #
######################
def load_nifti(path, normalize=True, downsample_factor=1.0):
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

def data_generator(pairs, batch_size, normalize=True, downsample_factor=1.0):
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

        inputs = [moving_batch, fixed_batch]
        shape = fixed_batch.shape[1:-1]
        zero_phi = np.zeros((batch_size, *shape, 3), dtype='float32')
        outputs = [fixed_batch, zero_phi]
        yield (inputs, outputs)

train_gen = data_generator(training_pairs, config.batch_size, normalize=config.normalize, downsample_factor=config.downsample_factor)
val_gen = data_generator(validation_pairs, config.batch_size, normalize=config.normalize, downsample_factor=config.downsample_factor)

#######################
# BUILD THE MODEL     #
#######################
# Infer shape from one volume
test_vol = load_nifti(training_pairs[0][0], normalize=config.normalize, downsample_factor=config.downsample_factor)
inshape = test_vol.shape[:-1]

print(f"Training image shape: {inshape}")

nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

vxm_model = vxm.networks.VxmDense(
    inshape=inshape,
    nb_unet_features=nb_features,
    int_steps=0
)

losses = [
    vxm.losses.NCC().loss,
    vxm.losses.Grad('l2').loss,
    # tf.keras.losses.MeanSquaredError(),
]
# loss_weights = [1.0, 0.01, 0.1]
loss_weights = [1.0, 0.01]


vxm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
    loss=losses,
    loss_weights=loss_weights
)

##########################
# MODEL CHECKPOINT & DIR #
##########################
# Create directory structure for model checkpoints
current_date = datetime.now().strftime("%Y-%m-%d")
date_dir = os.path.join(MODEL_DIR, current_date)
if not os.path.exists(date_dir):
    os.makedirs(date_dir)

existing_runs = [d for d in os.listdir(date_dir) if os.path.isdir(os.path.join(date_dir, d))]
next_run_number = len(existing_runs) + 1
run_dir = os.path.join(date_dir, f"run_{next_run_number:02d}")
os.makedirs(run_dir)

checkpoint_path = os.path.join(run_dir, 'voxelmorph_model_epoch_{epoch:02d}.h5')

##############################
# W&B CALLBACKS & TRAINING   #
##############################

# WandbMetricsLogger will log training & validation metrics
metrics_logger = WandbMetricsLogger()

# WandbModelCheckpoint will save models or weights to W&B as Artifacts.
# Here we save weights only when val_loss improves if validation is available
model_ckpt = WandbModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',  # assuming you want to monitor val_loss
    save_best_only=False,  # or True if you only want the best model
    save_weights_only=True
)

# Optional learning rate scheduler callback
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.5, patience=5, min_lr=1e-6
)

#################
# MODEL TRAIN   #
#################
vxm_model.fit(
    train_gen,
    steps_per_epoch=config.steps_per_epoch,
    epochs=config.epochs,
    validation_data=val_gen,
    validation_steps=len(validation_pairs)//config.batch_size if validation_pairs else None,
    callbacks=[metrics_logger, model_ckpt, lr_callback]
)

print(f"Model weights have been saved to: {run_dir}")
