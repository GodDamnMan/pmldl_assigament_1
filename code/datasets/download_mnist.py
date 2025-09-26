import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os





def download_and_preprocess_mnist():
    """Download and preprocess MNIST dataset"""
    print("Downloading MNIST dataset...")
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data for CNN (add channel dimension)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save raw data
    np.save('data/raw/x_train.npy', x_train)
    np.save('data/raw/y_train.npy', y_train)
    np.save('data/raw/x_test.npy', x_test)
    np.save('data/raw/y_test.npy', y_test)
    
    # Split training data for validation
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    # Save processed data
    np.save('data/processed/x_train.npy', x_train_split)
    np.save('data/processed/y_train.npy', y_train_split)
    np.save('data/processed/x_val.npy', x_val)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/x_test.npy', x_test)
    np.save('data/processed/y_test.npy', y_test)
    
    print("Dataset downloaded and preprocessed successfully!")
    print(f"Training samples: {len(x_train_split)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")

if __name__ == "__main__":
    download_and_preprocess_mnist()