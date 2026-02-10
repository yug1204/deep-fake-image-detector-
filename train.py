import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_dcnn_model
import os
import argparse

# --- Configuration ---
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.01

def train_model(data_dir, weights_output_path='dummy_weights.h5'):
    """
    Trains the D-CNN model.
    assumes data_dir structure:
    data_dir/
      train/
        REAL/
        DEEPFAKE/
      val/
        REAL/
        DEEPFAKE/
    """
    
    # 1. Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        print("Please structure your dataset as:")
        print("  dataset/train/REAL")
        print("  dataset/train/DEEPFAKE")
        print("  dataset/val/REAL")
        print("  dataset/val/DEEPFAKE")
        return

    print("Found training data. Setting up flow_from_directory...")
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # 2. Build Model
    print("Building D-CNN Model...")
    model = build_dcnn_model()
    model.summary()

    # 3. Callbacks
    checkpoint = ModelCheckpoint(
        weights_output_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )

    # 4. Train
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    print("Training finished.")
    print(f"Best weights saved to {weights_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train D-CNN Deepfake Detector")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset directory containing 'train' and 'val' folders")
    parser.add_argument('--output', type=str, default='dummy_weights.h5', help="Path to save output weights")
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output)
