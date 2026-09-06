from ultralytics import YOLO

# Load your custom trained model(you must first train the model so it appears in run folder)
model = YOLO('./runs/detect/banknote_cropper/weights/best.pt')

# Export to TFLite format
# This will create a folder named 'best_saved_model' containing 'best_float32.tflite'
model.export(format='tflite', imgsz=640)

print("Export complete. Grab the .tflite file!")