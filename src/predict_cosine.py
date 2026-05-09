import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import cosine


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/IRR/val/")
    parser.add_argument("--enc_path", type=str, default="./models/banknote_net_encoder.h5")
    parser.add_argument("--centroids_path", type=str, default="./src/trained_models/cosine_centroids.pkl")
    parser.add_argument("--threshold", type=float, default=0.80)
    return parser.parse_args()


def main():
    args = parse_arguments()
    IMG_SIZE = (224, 224)

    encoder = load_model(args.enc_path)
    with open(args.centroids_path, 'rb') as f:
        centroids = pickle.load(f)

    val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        args.data_path, target_size=IMG_SIZE, batch_size=1, shuffle=False, class_mode=None
    )

    embeddings = encoder.predict(val_gen, steps=val_gen.samples)

    print("\n" + "=" * 75 + "\nCOSINE PREDICTIONS (Handling 'None' folder)\n" + "=" * 75)
    correct = 0

    for i in range(val_gen.samples):
        full_path = val_gen.filenames[i].replace('\\', '/')
        actual_folder = full_path.split('/')[0]

        img_vector = embeddings[i]
        similarities = {name: 1 - cosine(img_vector, vec) for name, vec in centroids.items()}

        best_class = max(similarities, key=similarities.get)
        best_score = similarities[best_class]

        if best_score < args.threshold:
            pred_label = "NONE"
        else:
            pred_label = best_class

        # Accuracy Logic
        is_correct = False
        if actual_folder.lower() == "none":
            if pred_label == "NONE": is_correct = True
        else:
            if pred_label == actual_folder: is_correct = True

        status = "✓" if is_correct else "✗"
        if is_correct: correct += 1

        print(f"[{status}] {full_path:<40} -> Predicted: {pred_label:<18} (Sim: {best_score:.2f})")

    acc = (correct / val_gen.samples) * 100
    print(f"\nFinal Accuracy: {correct}/{val_gen.samples} ({acc:.2f}%)")


if __name__ == "__main__":
    main()