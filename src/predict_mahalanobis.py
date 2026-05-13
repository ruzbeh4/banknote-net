import argparse
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import mahalanobis


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/IRR/processed1/test/")
    parser.add_argument("--enc_path", type=str, default="./models/banknote_net_encoder.h5")
    parser.add_argument("--params_path", type=str, default="./src/trained_models/mahalanobis_data.pkl")
    parser.add_argument("--threshold", type=float, default=35.0)  # Adjust based on results
    return parser.parse_args()


def main():
    args = parse_arguments()
    encoder = load_model(args.enc_path)
    with open(args.params_path, 'rb') as f:
        data = pickle.load(f)

    centroids = data["centroids"]
    precision = data["precision"]

    test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        args.data_path, target_size=(224, 224), batch_size=1, shuffle=False, class_mode=None
    )

    embeddings = encoder.predict(test_gen)
    correct = 0

    print(f"\nMAHALANOBIS PREDICTIONS (Threshold: {args.threshold})\n" + "=" * 75)
    for i in range(test_gen.samples):
        actual_folder = test_gen.filenames[i].replace('\\', '/').split('/')[0]

        # Calculate Mahalanobis distance to each centroid
        dists = {name: mahalanobis(embeddings[i], ctr, precision) for name, ctr in centroids.items()}
        best_class = min(dists, key=dists.get)
        min_dist = dists[best_class]

        pred_label = "NONE" if min_dist > args.threshold else best_class

        is_correct = (pred_label == "NONE" and actual_folder.lower() == "none") or (pred_label == actual_folder)
        if is_correct: correct += 1

        print(f"[{'✓' if is_correct else '✗'}] {test_gen.filenames[i]:<40} -> {pred_label:<15} (Dist: {min_dist:.2f})")

    print(f"\nFinal Accuracy: {correct}/{test_gen.samples} ({(correct / test_gen.samples) * 100:.2f}%)")


if __name__ == "__main__":
    main()