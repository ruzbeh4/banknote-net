import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train the YOLO banknote detector.")
    parser.add_argument("--model_path", default="yolo26n.pt")
    parser.add_argument("--data_path", default="./data/yolo/data.yaml")
    return parser.parse_args()

# Put the execution logic inside this block!
if __name__ == '__main__':
    args = parse_args()
    # 1. Load the base pre-trained Nano model
    model = YOLO(args.model_path)

    # 2. Train it on your newly cleaned dataset
    model.train(
        data=args.data_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='banknote_cropper',
        exist_ok=True,

        # --- THE CUSTOM HYPERPARAMETERS ---
        # 1. Decrease classification importance (we only have 1 class anyway)
        cls=0.5,

        # 2. Double the penalty for bad box overlap
        # (Forces the model to care deeply about the 2:1 rectangle shape)
        box=15.0,

        # 3. Double the penalty for blurry/loose edges
        # (Forces the box to snap perfectly to the physical edges of the paper)
        dfl=3.0

    )

    print("Training complete. Check the 'runs/detect/banknote_cropper' folder for your new .pt and .tflite files.")