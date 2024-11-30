#!/usr/bin/env python3

import argparse
import os
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne

def parse_args():
    parser = argparse.ArgumentParser(description='Train VoxelMorph Model')
    parser.add_argument('--data-dir', required=True, help='Path to the training data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--model-dir', default='models', help='Directory to save models')
    parser.add_argument('--gpu', default='0', help='GPU ID to use')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load your data
    # Assuming your data is in NumPy format
    moving_images = np.load(os.path.join(args.data_dir, 'moving_images.npy'))
    fixed_images = np.load(os.path.join(args.data_dir, 'fixed_images.npy'))

    # Create data generators
    def data_generator(moving_images, fixed_images, batch_size):
        while True:
            idx = np.random.randint(0, len(moving_images), batch_size)
            moving_batch = moving_images[idx, ..., np.newaxis]
            fixed_batch = fixed_images[idx, ..., np.newaxis]
            inputs = [moving_batch, fixed_batch]
            outputs = [fixed_batch, np.zeros_like(moving_batch)]
            yield (inputs, outputs)

    train_gen = data_generator(moving_images, fixed_images, args.batch_size)

    # Define model architecture
    inshape = moving_images.shape[1:]  # Adjust based on your data
    nb_features = [
        [32, 64, 128, 256],
        [256, 128, 64, 32, 16]
    ]

    # Build VoxelMorph model
    vxm_model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_features=nb_features,
        int_steps=0  # Set to >0 for diffeomorphic transforms
    )

    # Compile the model
    losses = [
        vxm.losses.NCC().loss,
        vxm.losses.Grad('l2').loss
    ]
    weights = [1.0, 0.01]

    vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=weights)

    # Create model directory if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Set up callbacks (e.g., model checkpointing)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.model_dir, 'voxelmorph_model_{epoch:02d}.h5'),
        save_freq='epoch'
    )

    # Train the model
    vxm_model.fit(
        train_gen,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        callbacks=[checkpoint_callback]
    )

if __name__ == '__main__':
    main()
