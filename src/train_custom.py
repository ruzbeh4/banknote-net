"""
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the MIT License.

    Trains a model using images as input located in a custom folder and 
    the pre-trained banknote_net encoder network (MobileNet V2). Saves the best model 
    in ./src/trained_models/
"""

import argparse
import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_arguments():
    """Parses arguments for shallow classifier training."""
    parser = argparse.ArgumentParser(
        description="Train model from custom image folder using pre-trained BankNote-Net encoder."
    )
    parser.add_argument(
        "--bsize",
        "--b",
        type=int,
        help="Batch size",
        default=4,
    )
    parser.add_argument(
        "--epochs",
        "--e",
        type=int,
        help="Number of epochs for training shallow top classifier",
        default=25,
    )
    parser.add_argument(
        "--data_path",
        "--data",
        type=str,
        help="Path to folder with images.",
        default="./data/IRR/processed1",
    )
    parser.add_argument(
        "--enc_path",
        "--enc",
        type=str,
        help="Path to .h5 file of pre-trained encoder model",
        default="./models/banknote_net_encoder.h5",
    )

    return parser.parse_args()


def create_generator(
    TRAIN_PATH: str,
    VAL_PATH: str,
    IMG_SIZE: tuple,
    CLASS_NAMES: list,  # Added this to force strict class matching
    BATCH_SIZE: int = 2,
):
    """Creates tensorflow datasets for custom directory"""

    IMG_WIDTH, IMG_HEIGHT = IMG_SIZE
    NUM_CLASSES = len(CLASS_NAMES)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=180,
        channel_shift_range=40,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=12345,
        classes=CLASS_NAMES,  # Forces generator to only use the 12 training classes
        class_mode="categorical",
    )
    validation_generator = test_datagen.flow_from_directory(
        VAL_PATH,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=False,
        classes=CLASS_NAMES,  # Forces generator to ignore the 13th 'None' folder
        class_mode="categorical",
    )

    train_ds = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, IMG_HEIGHT, IMG_WIDTH, 3], [None, NUM_CLASSES]),
    )
    val_ds = tf.data.Dataset.from_generator(
        lambda: validation_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, IMG_HEIGHT, IMG_WIDTH, 3], [None, NUM_CLASSES]),
    )

    return train_ds, val_ds


def main():
    """Trains classifier for custom class and data directory."""

    args = parse_arguments()
    BATCH_SIZE = args.bsize
    NB_EPOCH = args.epochs
    ENC_PATH = args.enc_path
    DATA_PATH = args.data_path

    # Get the exact 12 class names from the train folder
    train_dir = os.path.join(DATA_PATH, "train")
    val_dir = os.path.join(DATA_PATH, "val")
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    NUM_CLASSES = len(class_names)

    IMG_SIZE = (224, 224)
    NB_TRAINING_SAMPLES = sum([len(files) for r, d, files in os.walk(train_dir)])
    # Only count validation files that belong to our 12 actual classes
    NB_VALIDATION_SAMPLES = sum([len(files) for r, d, files in os.walk(val_dir) if os.path.basename(r) in class_names])

    # Load datasets
    train_ds, val_ds = create_generator(
        TRAIN_PATH=train_dir,
        VAL_PATH=val_dir,
        IMG_SIZE=IMG_SIZE,
        CLASS_NAMES=class_names, # Pass the strict list of 12 classes
        BATCH_SIZE=BATCH_SIZE,
    )

    # Load encoder model and freeze layers
    encoder = load_model(ENC_PATH)
    for layer in encoder.layers:
        layer.trainable = False

    input_layer = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = encoder(input_layer)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.summary()

    # Create trained_models folder if it doesn't exist
    os.makedirs("./src/trained_models/", exist_ok=True)

    # Define callbacks, compile and fit
    checkpoint = ModelCheckpoint(
        filepath="./src/trained_models/custom_classifier.h5",
        monitor="val_acc",
        save_best_only=True,
    )

    # Compile and fit (Updated lr to learning_rate to avoid deprecation warning)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[
            "acc",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    model.fit(
        train_ds,
        steps_per_epoch=max(1, NB_TRAINING_SAMPLES // BATCH_SIZE),
        epochs=NB_EPOCH,
        validation_steps=max(1, NB_VALIDATION_SAMPLES // BATCH_SIZE),
        validation_data=val_ds,
        callbacks=[checkpoint],
    )


if __name__ == "__main__":
    main()