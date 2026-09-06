# Two-Stage Robust Offline Banknote Recognition Pipeline

> **Note:** This repository is an extension and fork of Microsoft's [BankNote-Net](https://github.com/microsoft/BankNote-Net). For the original dataset documentation, see [README_ORIGINAL.md](README_ORIGINAL.md).

An assistive, offline computer vision pipeline designed for resource-constrained edge devices (Android / TFLite). This project couples a spatial filtering stage (**YOLO26n**) with a fine-tuned **BankNote-Net** classifier to eliminate false positives on unconstrained live camera feeds.

---

## Benchmark Results (3,877 Test Images)

| Metric | Standalone BankNote-Net | Proposed Two-Stage Pipeline |
| :--- | :---: | :---: |
| **Total Test Images** | 3,877 | 3,877 |
| **Background Frames Evaluated** | 3,140 | 82 (3,058 filtered by YOLO) |
| **Background False Positive Rate** | 11.91% (374 errors) | **0.51%** (16 errors) |
| **Banknote Recognition (39 classes)** | 89.82% (662/737) | **88.87%** (655/737) |
| **Overall Pipeline Accuracy** | 88.42% | **97.47%** |

[//]: # (| **Inference Frame Rate** | ~10 FPS | **3–4 FPS** &#40;Galaxy S20 FE&#41; |)

---

## Environment Setup

Because YOLO and BankNote-Net have conflicting package requirements, the pipeline uses two isolated Conda environments:

### 1. YOLO Stage (`yolo_env`)
```bash
conda create -n yolo_env python=3.10 -y
conda activate yolo_env
pip install -r envs/requirements-yolo.txt
```

### 2. Banknote-net Stage (`banknotenet_env`)
```bash
conda create -n banknote_env python=3.10 -y
conda activate banknote_env
pip install -r envs/requirements-banknote.txt
```

The commands below assume they are run from the repository root. Do not mix the
two environments: YOLO scripts use `yolo_env`, while BankNote-Net, plotting, and
the H5-to-TFLite classifier converter use `banknote_env`.

## First: Compare the Two Inference Paths

The fastest way to reproduce the comparison is to run the same classifier and
weights against the two prepared IRR datasets. `predict_custom.py` reads the
`test/` directory inside the path supplied by `--data_path`; it gets the class
names from that dataset's `train/` directory.

### Standalone BankNote-Net

This evaluates the original processed data, including its background/`None`
examples:

```bash
conda activate banknote_env
python src/predict_custom.py --data_path ./data/IRR/processed --model_path ./src/trained_models/custom_classifier.h5 --threshold 0.7
```

### YOLO + BankNote-Net

This evaluates the version filtered by the YOLO cropper:

```bash
conda activate banknote_env
python src/predict_custom.py --data_path ./data/IRR/processed-filtered-by-yolo --model_path ./src/trained_models/custom_classifier.h5 --threshold 0.7
```

Change `--threshold` to test a different confidence cutoff. Predictions below
the cutoff are reported as `NONE`. The script prints overall accuracy and
accuracy on banknote images without background noise.

## YOLO Dataset and Training

> **Notice:** The guides below use the default parameter values. You can adjust
> them as needed, or run the scripts with their defaults without specifing with flags to reproduce the
> documented results.

The YOLO dataset is not included in this repository. Put your own YOLO-format
dataset in a location you control and update `data/yolo/data.yaml` so its
`path`, `train`, `val`/`valid`, and `test` entries point to that dataset. The
dataset should contain the usual image and label folders, with one label file
per image in YOLO bounding-box format:

```text
class_id center_x center_y width height
```

For this project, use one class named `banknote`. If your source dataset has
separate currency or denomination classes, first edit the paths in
`src/yolo/clean_dataset_with_only_one_banknote_class.py` and run it from the
repository root. It changes the first value of every annotation to `0`,
turning all annotated banknote classes into the single class expected by the
cropper. Make a backup before running it because it edits label files in place.

Train the nano detector in the YOLO environment:

```bash
conda activate yolo_env
python src/yolo/train_yolo.py
```

The script uses `yolo26n.pt`, `data/yolo/data.yaml`, 50 epochs, 640-pixel
images, and batch size 16. The best weights are written to
`runs/detect/banknote_cropper/weights/best.pt`.

## Run YOLO Inference and Crop Images

`src/yolo/yolo.py` is a small folder-based inference script. Before running it,
check these values near the top of the file:

```python
custom_model_path = "./runs/detect/banknote_cropper/weights/best.pt"
input_folder = "./data/crop_test/input"
output_folder_debug = "./data/crop_test/output_debug"
output_folder_cropped = "./data/crop_test/output_cropped"
```

Put full camera images in `data/crop_test/input`, then run:

```bash
conda activate yolo_env
python src/yolo/yolo.py
```

The script saves every annotated image to `output_debug`. When it finds a
largest detection covering more than 30% of the source image, it saves the
corresponding clean crop to `output_cropped`. Images without a passing box are
kept only in the debug output.

## Move Crops into BankNote-Net Folders

BankNote-Net expects a directory tree whose immediate subfolders are class
names, for example:

```text
data/IRR/processed-filtered-by-yolo/
	train/<class-name>/...
	test/<class-name>/...
```

`src/image_resizer.py` recursively copies images to a new destination,
converts them to RGB, applies EXIF rotation, and resizes them to `224x224`.
It is configured by editing the two path assignments at the bottom of the
file, then running it from the repository root:

```bash
conda activate banknote_env
python src/image_resizer.py
```

Set `input_dir` to the YOLO crop directory and `output_dir` to the exact class
folder where those crops belong, for example
`./data/IRR/processed-filtered-by-yolo/test/None` for background crops. The
current default output uses the spelling `processed-filteredByYolo`; change it
if the intended dataset folder is `processed-filtered-by-yolo`.

The current `output_cropped` input is full of `None`-class data rather than
real banknotes. Sort the crops into the correct denomination/currency folders
before using them as banknote training or evaluation data. Keep genuine
background images in the `None` folder. `src/image_resizer_2.py` is a separate
in-place utility that pads images to a square with black borders and resizes
them to `224x224`; edit its `target_dir` before using it.

## Train and Use the Custom BankNote-Net Classifier

The classifier uses the pre-trained encoder at
`models/banknote_net_encoder.h5`. Put each class in a subfolder of
`<data_path>/train`; `train_custom.py` creates an 80/20 split from those
folders, freezes the encoder, and trains a shallow classifier:

```bash
conda activate banknote_env
python src/train_custom.py --data_path ./data/IRR/processed --enc_path ./models/banknote_net_encoder.h5 --epochs 40 --bsize 32
```

The best model is saved as `src/trained_models/custom_classifier.h5`.
Use a different `--data_path` to train on the YOLO-filtered dataset, provided
that its `train/` folders contain correctly sorted classes. For inference,
use the matching dataset root with `src/predict_custom.py`, as shown above.

To compare standalone BankNote-Net with the two-stage pipeline, switch only
`--data_path` between:

```text
./data/IRR/processed
./data/IRR/processed-filtered-by-yolo
```

Both roots must contain compatible `train/` and `test/` directory structures.
The classifier does not detect a banknote in the scene; YOLO performs that
spatial filtering before the cropped image reaches the classifier.

## BankNote-Net Analysis Utilities

These utilities belong to `banknote_env`:

### Latent-space t-SNE

`src/plot_tsne.py` extracts encoder embeddings from `train/` and `test/` and
saves a two-dimensional plot. Change `--data_path` to compare the unfiltered
and YOLO-filtered datasets:

```bash
conda activate banknote_env
python src/plot_tsne.py --data_path ./data/IRR/processed --enc_path ./models/banknote_net_encoder.h5 --save_plot ./tsne_processed.png
python src/plot_tsne.py --data_path ./data/IRR/processed-filtered-by-yolo --enc_path ./models/banknote_net_encoder.h5 --save_plot ./tsne_filtered.png
```

### Confidence-threshold plot

`src/plot_effect_of_threshold.py` runs `predict_custom.py` repeatedly for
thresholds from `0.55` through `0.95` and plots overall and banknote-only
accuracy. It uses the paths and model defaults from `predict_custom.py`; edit
`TARGET_SCRIPT` or the classifier defaults when comparing another dataset.
Run it only when you intentionally want all of those repeated evaluations:

```bash
conda activate banknote_env
python src/plot_effect_of_threshold.py
```

The output is `threshold_evaluation_plot.png`.

## Export Models to TFLite

### BankNote-Net classifier

Edit `address` in `src/converter.py` to the base path of the classifier H5
file. For the included custom classifier it should be:

```python
address = './src/trained_models/custom_classifier'
```

Then run the converter in `banknote_env`:

```bash
conda activate banknote_env
python src/converter.py
```

It writes the quantized/default-optimized TFLite model beside the H5 file.

### YOLO detector

After training, confirm the weights path in `src/yolo/converter.py` and run it
in `yolo_env`:

```bash
conda activate yolo_env
python src/yolo/converter.py
```

Ultralytics exports the detector to a `best_saved_model` directory containing
the TFLite artifact. Copy the resulting YOLO and classifier TFLite files into
the Android application and keep their preprocessing, input size, output
format, class order, and confidence threshold consistent with the Python
pipeline.

## Main Files

| Purpose | File |
| --- | --- |
| YOLO training | [`src/yolo/train_yolo.py`](src/yolo/train_yolo.py) |
| YOLO inference and cropping | [`src/yolo/yolo.py`](src/yolo/yolo.py) |
| Squash/copy crops to `224x224` | [`src/image_resizer.py`](src/image_resizer.py) |
| Train custom classifier | [`src/train_custom.py`](src/train_custom.py) |
| Classifier inference | [`src/predict_custom.py`](src/predict_custom.py) |
| t-SNE analysis | [`src/plot_tsne.py`](src/plot_tsne.py) |
| Threshold analysis | [`src/plot_effect_of_threshold.py`](src/plot_effect_of_threshold.py) |
| Classifier H5 to TFLite | [`src/converter.py`](src/converter.py) |
| YOLO weights to TFLite | [`src/yolo/converter.py`](src/yolo/converter.py) |
| Collapse YOLO labels to one class | [`src/yolo/clean_dataset_with_only_one_banknote_class.py`](src/yolo/clean_dataset_with_only_one_banknote_class.py) |

