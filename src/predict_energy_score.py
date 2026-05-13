import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/IRR/processed1")
    parser.add_argument("--model_path", type=str, default="./src/trained_models/custom_classifier_energy.h5")
    parser.add_argument("--stats_path", type=str, default="./src/trained_models/energy_stats.pkl")
    # You can override the suggest threshold here
    parser.add_argument("--threshold", type=float, default=-4)
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load stats and set threshold
    with open(args.stats_path, 'rb') as f:
        stats = pickle.load(f)
    energy_threshold = args.threshold if args.threshold is not None else stats['suggested_threshold']

    # Get labels
    train_dir = os.path.join(args.data_path, "train")
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    index_to_class = {i: name for i, name in enumerate(class_names)}

    # Build Logit Model
    full_model = load_model(args.model_path)
    # Layer -2 is the Dense layer before the Activation layer
    logit_model = Model(inputs=full_model.input, outputs=full_model.get_layer("logits_layer").output)

    test_dir = os.path.join(args.data_path, "test")
    test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        test_dir, target_size=(224, 224), batch_size=1, shuffle=False, class_mode=None
    )

    logits = logit_model.predict(test_gen)
    T = 1.0
    energies = -T * np.log(np.sum(np.exp(logits / T), axis=1))

    # Softmax probabilities for final classification
    probs = tf.nn.softmax(logits).numpy()

    print(f"\nENERGY PREDICTIONS (Threshold: {energy_threshold:.2f})\n" + "=" * 75)
    correct = 0

    for i in range(test_gen.samples):
        actual_folder = test_gen.filenames[i].replace('\\', '/').split('/')[0]
        energy = energies[i]

        if energy > energy_threshold:
            pred_label = "NONE"
        else:
            pred_label = index_to_class[np.argmax(probs[i])]

        is_correct = (pred_label == "NONE" and actual_folder.lower() == "none") or (pred_label == actual_folder)
        if is_correct: correct += 1

        status = "✓" if is_correct else "✗"
        print(f"[{status}] {test_gen.filenames[i]:<40} -> {pred_label:<18} (Energy: {energy:.2f})")

    print(f"\nFinal Accuracy: {correct}/{test_gen.samples} ({(correct / test_gen.samples) * 100:.2f}%)")


if __name__ == "__main__":
    main()