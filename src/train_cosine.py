import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/IRR/train/")
    parser.add_argument("--enc_path", type=str, default="./models/banknote_net_encoder.h5")
    parser.add_argument("--save_path", type=str, default="./src/trained_models/cosine_centroids.pkl")
    return parser.parse_args()


def main():
    args = parse_arguments()
    IMG_SIZE = (224, 224)

    print("Loading Encoder...")
    encoder = load_model(args.enc_path)

    print("Loading Training Data to calculate embeddings...")
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_gen = train_datagen.flow_from_directory(
        args.data_path, target_size=IMG_SIZE, batch_size=1, shuffle=False, class_mode="categorical"
    )

    print("Extracting features (this may take a minute)...")
    # Extract 1280-dimensional feature vectors for every image
    embeddings = encoder.predict(train_gen, steps=train_gen.samples)

    # Calculate the mean vector (centroid) for each class
    centroids = {}
    index_to_class = {v: k for k, v in train_gen.class_indices.items()}

    for class_idx in range(len(train_gen.class_indices)):
        # Find all image indices belonging to this class
        class_image_indices = np.where(train_gen.classes == class_idx)[0]
        # Get their embeddings
        class_embeddings = embeddings[class_image_indices]
        # Average them to create the prototype
        centroids[index_to_class[class_idx]] = np.mean(class_embeddings, axis=0)

    # Save the centroids
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'wb') as f:
        pickle.dump(centroids, f)

    print(f"\nSaved {len(centroids)} class centroids to {args.save_path}")


if __name__ == "__main__":
    main()