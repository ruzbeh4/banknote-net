import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsize", type=int, default=1)
    # switch to processed-filtered-by-yolo to see pipeline of yolo+banknote-net results
    parser.add_argument("--data_path", type=str, default="./data/IRR/processed", help="Path to IRR folder containing train/ and test/")
    parser.add_argument("--model_path", type=str, default="./src/trained_models/custom_classifier.h5")
    parser.add_argument("--threshold", type=float, default=0.7)
    return parser.parse_args()

def main():
    args = parse_arguments()
    IMG_SIZE = (224, 224)

    # SOURCE OF TRUTH: Get classes from the TRAIN folder (always 12)
    train_dir = os.path.join(args.data_path, "train")
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    NUM_CLASSES = len(class_names)
    index_to_class = {i: name for i, name in enumerate(class_names)}

    # DATA TO TEST: Look at the test folder (can have 13+ folders now)
    test_dir = os.path.join(args.data_path, "test")
    test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=1, shuffle=False, class_mode=None # class_mode=None prevents label mismatch
    )

    model = load_model(args.model_path)
    preds = model.predict(test_gen, steps=test_gen.samples)

    print("\n" + "="*75 + "\nNN PREDICTIONS (Handling 'None' folder)\n" + "="*75)
    correct = 0
    corrects_without_background_noise = 0
    total_banknotes = 0

    for i in range(test_gen.samples):
        # Identify the actual folder name
        full_path = test_gen.filenames[i].replace('\\', '/')
        actual_folder = full_path.split('/')[0]

        img_probs = preds[i]
        top_idx = np.argmax(img_probs)
        conf = img_probs[top_idx]

        # Logic for "None" detection
        if conf < args.threshold:
            pred_label = "NONE"
        else:
            pred_label = index_to_class[top_idx]

        # Accuracy Logic:
        # If actual folder is 'None', prediction must be 'NONE' to be correct.
        # If actual folder is a banknote, prediction must match that folder name.
        is_correct = False
        if actual_folder.lower() == "none":
            if pred_label == "NONE" or pred_label == "None": is_correct = True
        else:
            if pred_label == actual_folder:
                is_correct = True
                corrects_without_background_noise += 1
            total_banknotes += 1


        status = "✓" if is_correct else "✗"
        if is_correct: correct += 1

        print(f"[{status}] {full_path:<40} -> Predicted: {pred_label:<18} (Conf: {conf:.2f})")

    acc = (correct / test_gen.samples) * 100
    banknote_acc = (corrects_without_background_noise / total_banknotes) * 100
    print(f"\nFinal Accuracy: {correct}/{test_gen.samples} ({acc:.2f}%)")
    print(f"\nAccuracy without background noise: {corrects_without_background_noise}/{total_banknotes} ({banknote_acc:.2f}%)")

if __name__ == "__main__":
    main()