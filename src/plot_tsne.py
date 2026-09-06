import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot t-SNE of Banknote-Net Latent Space")
    parser.add_argument("--data_path", type=str, default="./data/IRR/processed1")
    parser.add_argument("--enc_path", type=str, default="./models/banknote_net_encoder.h5")
    parser.add_argument("--save_plot", type=str, default="./tsne_latent_space.png")
    return parser.parse_args()


def get_embeddings_and_labels(encoder, folder_path):
    if not os.path.exists(folder_path):
        return np.array([]), np.array([])

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_directory(
        folder_path, target_size=(224, 224), batch_size=1, shuffle=False
    )

    if gen.samples == 0:
        return np.array([]), np.array([])

    print(f"Extracting features from {folder_path}...")
    embeddings = encoder.predict(gen, steps=gen.samples, verbose=1)

    # Map numerical indices back to string class names (e.g., '100000_1_back')
    index_to_class = {v: k for k, v in gen.class_indices.items()}
    labels = [index_to_class[gen.classes[i]] for i in range(gen.samples)]

    return embeddings, np.array(labels)


def main():
    args = parse_arguments()

    print("Loading Encoder...")
    encoder = load_model(args.enc_path)

    subdirs = ["train", "test"]
    all_embeddings = []
    all_labels = []

    for subdir in subdirs:
        full_path = os.path.join(args.data_path, subdir)
        emb, lab = get_embeddings_and_labels(encoder, full_path)
        if len(emb) > 0:
            all_embeddings.append(emb)
            all_labels.append(lab)

    if not all_embeddings:
        print("No images found in the specified data paths.")
        return

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)

    print(f"\nTotal images processed: {len(all_labels)}")
    print("Running t-SNE dimensionality reduction (this may take a moment)...")

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(all_embeddings)

    print("Plotting results...")
    plt.figure(figsize=(14, 10))

    # Create a distinct color palette
    unique_classes = np.unique(all_labels)
    palette = sns.color_palette("husl", len(unique_classes))
    color_dict = dict(zip(unique_classes, palette))

    # Force the 'None' class to be Black with a larger 'X' marker so it stands out
    if 'None' in color_dict:
        color_dict['None'] = '#000000'

    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=all_labels,
        palette=color_dict,
        alpha=0.8,
        s=80,
        edgecolor='w'
    )

    plt.title("t-SNE Projection of Banknote-Net Latent Space (1280D -> 2D)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    plt.savefig(args.save_plot, dpi=300)
    print(f"Plot saved successfully to {args.save_plot}")
    plt.show()


if __name__ == "__main__":
    main()