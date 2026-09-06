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
        default=32,  # Restored to original default
    )
    parser.add_argument(
        "--epochs",
        "--e",
        type=int,
        help="Number of epochs for training shallow top classifier",
        default=40,  # Restored to original default
    )
    parser.add_argument(
        "--data_path",
        "--data",
        type=str,
        help="Path to folder with images.",
        default="./data/IRR/processed",
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
    DATA_DIR: str,
    IMG_SIZE: tuple,
    BATCH_SIZE: int,
):
    """Creates tensorflow datasets utilizing inner Keras validation split."""

    IMG_WIDTH, IMG_HEIGHT = IMG_SIZE
    SPLIT_SEED = 12345  # Locks the split so train and val don't overlap

    # Training Generator (Includes augmentation + 20% split directive)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=180,
        channel_shift_range=40,
        fill_mode="nearest",
        validation_split=0.2,
    )

    # Validation Generator (NO augmentation, strictly shares the same split directive)
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SPLIT_SEED,
        class_mode="categorical",
        subset="training",
    )

    validation_generator = test_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SPLIT_SEED,
        class_mode="categorical",
        subset="validation",
    )

    NUM_CLASSES = train_generator.num_classes

    # Returning raw generators directly prevents tf.data bottlenecking
    return train_generator, validation_generator, train_generator.samples, validation_generator.samples, NUM_CLASSES


def main():
    """Trains classifier for custom class and data directory."""

    args = parse_arguments()
    BATCH_SIZE = args.bsize
    NB_EPOCH = args.epochs
    ENC_PATH = args.enc_path
    DATA_PATH = args.data_path

    # Point directly to the train folder
    train_dir = os.path.join(DATA_PATH, "train")
    IMG_SIZE = (224, 224)

    # Load generators
    train_gen, val_gen, NB_TRAINING_SAMPLES, NB_VALIDATION_SAMPLES, NUM_CLASSES = create_generator(
        DATA_DIR=train_dir,
        IMG_SIZE=IMG_SIZE,
        BATCH_SIZE=BATCH_SIZE,
    )

    print(f"\nFound {NB_TRAINING_SAMPLES} training samples and {NB_VALIDATION_SAMPLES} validation samples across {NUM_CLASSES} classes.\n")

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

    os.makedirs("./src/trained_models/", exist_ok=True)

    checkpoint = ModelCheckpoint(
        filepath="./src/trained_models/custom_classifier.h5",
        monitor="val_acc",
        save_best_only=True,
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[
            "acc",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    # Fit using stable raw generators
    model.fit(
        train_gen,
        steps_per_epoch=max(1, NB_TRAINING_SAMPLES // BATCH_SIZE),
        epochs=NB_EPOCH,
        validation_data=val_gen,
        validation_steps=max(1, NB_VALIDATION_SAMPLES // BATCH_SIZE),
        callbacks=[checkpoint],
    )


if __name__ == "__main__":
    main()