import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train NN with Energy Score monitoring.")
    parser.add_argument("--bsize", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--data_path", type=str, default="./data/IRR/processed1")
    parser.add_argument("--enc_path", type=str, default="./models/banknote_net_encoder.h5")
    # Save with a new name to avoid overriding original
    parser.add_argument("--model_save", type=str, default="./src/trained_models/custom_classifier_energy.h5")
    return parser.parse_args()


def main():
    args = parse_arguments()
    IMG_SIZE = (224, 224)

    # 1. Setup Data (Strictly 12 classes)
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    num_classes = len(class_names)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, rotation_range=180, channel_shift_range=40, fill_mode="nearest"
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=args.bsize, classes=class_names, class_mode="categorical"
    )
    val_gen = test_datagen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=args.bsize, classes=class_names, class_mode="categorical"
    )

    # 2. Build Original NN Architecture
    encoder = load_model(args.enc_path)
    for layer in encoder.layers: layer.trainable = False

    input_layer = Input(shape=(224, 224, 3))
    x = encoder(input_layer)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    # The logits are here (last layer before softmax)
    logits_layer = Dense(num_classes, name="logits_layer")(x)
    output_layer = tf.keras.layers.Activation("softmax")(logits_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["acc"])

    # 3. Train
    os.makedirs("./src/trained_models/", exist_ok=True)
    checkpoint = ModelCheckpoint(filepath=args.model_save, monitor="val_acc", save_best_only=True)

    print("\nStarting NN Training (Energy Variant)...")
    model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=[checkpoint])

    # 4. Energy Analysis (Post-Training)
    print("\nCalculating Baseline Energy Scores for Thresholding...")
    # Create a sub-model to get the raw logits
    logit_model = Model(inputs=model.input, outputs=model.get_layer("logits_layer").output)

    # We use non-augmented data for statistics
    stat_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=1, shuffle=False, classes=class_names, class_mode=None
    )

    logits = logit_model.predict(stat_gen)
    # Energy = -T * log(sum(exp(logits/T)))
    T = 1.0
    energies = -T * np.log(np.sum(np.exp(logits / T), axis=1))

    stats = {
        "mean_energy": np.mean(energies),
        "max_energy": np.max(energies),
        "suggested_threshold": np.percentile(energies, 95)  # 95th percentile of banknotes
    }

    with open("./src/trained_models/energy_stats.pkl", 'wb') as f:
        pickle.dump(stats, f)

    print(f"\nTraining Complete. Model: {args.model_save}")
    print(f"Suggested Energy Threshold: {stats['suggested_threshold']:.2f} (Reject if > this)")


if __name__ == "__main__":
    main()