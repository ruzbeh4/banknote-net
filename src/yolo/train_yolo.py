from ultralytics import YOLO

# Put the execution logic inside this block!
if __name__ == '__main__':
    # 1. Load the base pre-trained Nano model
    model = YOLO('yolo26n.pt')

    # 2. Train it on your newly cleaned dataset
    model.train(
        data='./data/yolo/data.yaml',
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