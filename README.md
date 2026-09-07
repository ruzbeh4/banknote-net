# Two-Stage Robust Offline Banknote Recognition Pipeline

> **Note:** This repository is an extension and fork of Microsoft's [BankNote-Net](https://github.com/microsoft/BankNote-Net). For the original dataset documentation, see [README_ORIGINAL.md](README_ORIGINAL.md).

An assistive, offline computer vision pipeline designed for resource-constrained edge devices (Android / TFLite). This project couples a spatial filtering stage (**YOLO26n**) with a fine-tuned **BankNote-Net** classifier to eliminate false positives on unconstrained live camera feeds.

---

## Benchmark Results (4,498 Test Images)

| Metric | Standalone BankNote-Net (conf = 0.70) | Proposed Two-Stage Pipeline (conf_yolo = 0.35, conf_bn = 0.40) |
| :--- | :---: | :---: |
| **Total Test Images** | 4,498 | 4,498 |
| **Classifier Invocations** | 4,498 | **1,269** (71.79% reduction) |
| **Background Frames Evaluated** | 3,219 | **87** (3,132 filtered by YOLO) |
| **Background False Positive Rate** | 11.77% (379 errors) | **0.90%** (29 errors) |
| **Banknote Recognition (39 classes)** | 84.75% (1,084/1,279) | **82.64%** (1,057/1,279) |
| **Overall Pipeline Accuracy** | 87.24% (3,924/4,498) | **94.42%** (4,247/4,498) |

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
python src/predict_custom.py --data_path ./data/IRR/processed-filtered-by-yolo --model_path ./src/trained_models/custom_classifier.h5 --threshold 0.4
```
Note that we lower banknote-net threshold duo to successful removal of almost all background images.
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
separate currency or denomination classes, run
`src/yolo/clean_dataset_with_only_one_banknote_class.py` with one or more label
directories. It changes the first value of every annotation to `0`, turning
all annotated banknote classes into the single class expected by the cropper.
Make a backup before running it because it edits label files in place.

Train the nano detector in the YOLO environment:

```bash
conda activate yolo_env
python src/yolo/clean_dataset_with_only_one_banknote_class.py --label_folders ./data/yolo/train/labels ./data/yolo/valid/labels ./data/yolo/test/labels
python src/yolo/train_yolo.py --model_path ./yolo26n.pt --data_path ./data/yolo/data.yaml
```

The default values are `yolo26n.pt` and `./data/yolo/data.yaml`, so the
training command can also be simply `python src/yolo/train_yolo.py`. Training
uses 50 epochs, 640-pixel images, and batch size 16. The best weights are
written to `runs/detect/banknote_cropper/weights/best.pt`.

## Run YOLO Inference and Crop Images

`src/yolo/yolo.py` is a folder-based inference script. Its path defaults are:

```text
--model_path ./runs/detect/banknote_cropper/weights/best.pt
--input_folder ./data/crop_test/input
--output_folder_debug ./data/crop_test/output_debug
--output_folder_cropped ./data/crop_test/output_cropped
```

Put full camera images in `data/crop_test/input`, then run:

```bash
conda activate yolo_env
python src/yolo/yolo.py --model_path ./runs/detect/banknote_cropper/weights/best.pt --input_folder ./data/crop_test/input --output_folder_debug ./data/crop_test/output_debug --output_folder_cropped ./data/crop_test/output_cropped
```

The same command without flags uses these defaults. It saves every annotated
image to `output_debug`. When it finds a largest detection covering more than
30% of the source image, it saves the corresponding clean crop to
`output_cropped`. Images without a passing box are kept only in the debug
output.

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
Pass the source and destination with `--input_dir` and `--output_dir`:

```bash
conda activate banknote_env
python src/image_resizer.py --input_dir ./data/crop_test/output_cropped --output_dir ./data/IRR/processed-filteredByYolo/test/None
```

The defaults are the YOLO crop directory and
`./data/IRR/processed-filteredByYolo/test/None`. Set `--output_dir` to the
exact class folder where the crops belong, for example
`./data/IRR/processed-filtered-by-yolo/test/None` for background crops.

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
saves a two-dimensional plot. Run it once for the original BankNote-Net data
and once for the YOLO-filtered pipeline data:

```bash
conda activate banknote_env
python src/plot_tsne.py --data_path ./data/IRR/processed --enc_path ./models/banknote_net_encoder.h5 --save_plot ./tsne_before_yolo.jpg
python src/plot_tsne.py --data_path ./data/IRR/processed-filtered-by-yolo --enc_path ./models/banknote_net_encoder.h5 --save_plot ./tsne-after-yolo.png
```

The first command produces the pure BankNote-Net latent-space plot;
the second produces the YOLO + BankNote-Net pipeline plot.

### Confidence-threshold plot

`src/plot_effect_of_threshold.py` runs `predict_custom.py` repeatedly for
thresholds from `0.55` through `0.95` and plots overall and banknote-only
accuracy. Use `--target_script` to select the classifier script and
`--output_plot` to choose the output image. Run it only when you intentionally
want all of those repeated evaluations:

```bash
conda activate banknote_env
python src/plot_effect_of_threshold.py --target_script ./src/predict_custom.py --output_plot threshold_evaluation_plot.png
```

The output is the tracked root-level file `threshold_evaluation_plot.png`.

## Export Models to TFLite

`src/converter.py` converts the BankNote-Net classifier from H5 to an optimized TFLite model.
`src/yolo/converter.py` exports the trained YOLO detector to TFLite using Ultralytics.

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

