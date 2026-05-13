import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/IRR/processed1/train/")
    parser.add_argument("--enc_path", type=str, default="./models/banknote_net_encoder.h5")
    parser.add_argument("--save_path", type=str, default="./src/trained_models/mahalanobis_data.pkl")
    return parser.parse_args()


def main():
    args = parse_arguments()
    encoder = load_model(args.enc_path)

    train_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        args.data_path, target_size=(224, 224), batch_size=1, shuffle=False
    )

    embeddings = encoder.predict(train_gen)

    centroids = {}
    centered_list = []

    for class_idx, class_name in {v: k for k, v in train_gen.class_indices.items()}.items():
        indices = np.where(train_gen.classes == class_idx)[0]
        class_embeddings = embeddings[indices]
        mean_vec = np.mean(class_embeddings, axis=0)
        centroids[class_name] = mean_vec
        # Subtract mean to calculate shared covariance later
        centered_list.append(class_embeddings - mean_vec)

    # Compute shared inverse covariance matrix
    all_centered = np.vstack(centered_list)
    covariance = np.cov(all_centered, rowvar=False)
    # Using Pseudo-Inverse for stability
    precision = np.linalg.pinv(covariance)

    save_data = {"centroids": centroids, "precision": precision}
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"Mahalanobis training complete. Parameters saved to {args.save_path}")


if __name__ == "__main__":
    main()