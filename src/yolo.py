import cv2
import os
from ultralytics import YOLO

# 1. Automatically downloads the tiny YOLOv8 model weights
model = YOLO("yolov8n.pt")

input_folder = "./data/crop_test/input"
output_folder = "./data/crop_test/output_debug"  # Changed to a debug folder
os.makedirs(output_folder, exist_ok=True)

# 2. Run inference with the 224 constraint you want to test
results = model(input_folder, stream=True)

for i, result in enumerate(results):
    # Make a copy of the image so we can draw shapes on it
    img_drawn = result.orig_img.copy()
    img_h, img_w = img_drawn.shape[:2]
    total_image_area = img_h * img_w

    max_area = 0
    best_box = None

    # 3. First pass: Find the biggest box that meets the > 50% rule
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_area = (x2 - x1) * (y2 - y1)
        area_percentage = box_area / total_image_area

        if box_area > max_area and area_percentage > 0.50:
            max_area = box_area
            best_box = (x1, y1, x2, y2)

    # 4. Second pass: Draw ALL boxes on the image so you can debug the results
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Calculate area again for the label
        box_area = (x2 - x1) * (y2 - y1)
        area_percentage = (box_area / total_image_area) * 100

        # Get the label name (e.g., "person", "book", "vase")
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Check if this specific box is our "winner"
        is_best = best_box and (x1, y1, x2, y2) == best_box

        if is_best:
            color = (0, 255, 0)  # Green for passing
            thickness = 3
            status = f"PASS ({area_percentage:.1f}%)"
        else:
            color = (0, 0, 255)  # Red for failing
            thickness = 1
            status = f"FAIL ({area_percentage:.1f}%)"

        # Draw the rectangle
        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, thickness)

        # Add the text label above the box
        text = f"{label} | {status}"
        cv2.putText(img_drawn, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 5. Save the annotated image (not cropped) to see the results
    save_path = os.path.join(output_folder, f"annotated_test_{i}.jpg")
    cv2.imwrite(save_path, img_drawn)

    if best_box:
        print(f"Image {i}: Found passing object. Saved {save_path}")
    else:
        print(f"Image {i}: No passing object. Saved {save_path} to view detections.")