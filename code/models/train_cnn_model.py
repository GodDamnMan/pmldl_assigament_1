import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_cnn_model():
    """Create a CNN model for MNIST digit classification"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def train_model():
    """Train the CNN model on MNIST data"""
    print("Loading preprocessed data...")
    
    # Load processed data
    x_train = np.load('data/processed/x_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    x_val = np.load('data/processed/x_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    
    # Create model
    model = create_cnn_model()
    
    # Compile model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    print("Model architecture:")
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.save('models/mnist_cnn_model.h5')
    print("Model saved successfully!")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("Training history plot saved!")
    
    # Evaluate on test data
    x_test = np.load('data/processed/x_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    train_model()