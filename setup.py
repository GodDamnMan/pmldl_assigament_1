#!/usr/bin/env python3
"""
Setup script for MNIST Digit Classification Project
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def main():
    print("Setting up MNIST Digit Classification Project")
    print("=" * 50)
    
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed', 
        'models',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created")
    
    # Download and preprocess data
    if run_command("python code/datasets/download_mnist.py", "Downloading MNIST dataset"):
        print("Data downloaded and preprocessed")
    else:
        print("Failed to download data")
        sys.exit(1)
    
    # Train the model
    if run_command("python code/models/train_cnn_model.py", "Training CNN model"):
        print("Model trained and saved")
    else:
        print("Failed to train model")
        sys.exit(1)
    
    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main()