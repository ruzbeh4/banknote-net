import cv2
import os
from ultralytics import YOLO

# 1. Automatically load your custom trained weights
custom_model_path = "./runs/detect/banknote_cropper/weights/best.pt"
model = YOLO(custom_model_path)

print("\n--- MODEL VERIFICATION ---")
print(f"Classes known by this model: {model.names}")
print("--------------------------\n")

input_folder = "./data/crop_test/input"
output_folder_debug = "./data/crop_test/output_debug"
output_folder_cropped = "./data/crop_test/output_cropped"

# Create both directories if they don't exist
os.makedirs(output_folder_debug, exist_ok=True)
os.makedirs(output_folder_cropped, exist_ok=True)

# 2. Run inference
results = model(input_folder, stream=True)

for i, result in enumerate(results):
    # Make a copy of the image so we can draw shapes on it later
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

        if box_area > max_area and area_percentage > 0.30:
            max_area = box_area
            best_box = (x1, y1, x2, y2)

    # NEW: 4. Crop and save the clean image if a valid box was found
    if best_box:
        cx1, cy1, cx2, cy2 = best_box
        # We slice from the original image to ensure no bounding box lines are included
        cropped_img = result.orig_img[cy1:cy2, cx1:cx2]
        crop_save_path = os.path.join(output_folder_cropped, f"cropped_{i}.jpg")
        cv2.imwrite(crop_save_path, cropped_img)

    # 5. Second pass: Draw ALL boxes on the debug image copy
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        box_area = (x2 - x1) * (y2 - y1)
        area_percentage = (box_area / total_image_area) * 100

        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        is_best = best_box and (x1, y1, x2, y2) == best_box

        if is_best:
            color = (0, 255, 0)  # Green for passing
            thickness = 3
            status = f"PASS ({area_percentage:.1f}%)"
        else:
            color = (0, 0, 255)  # Red for failing
            thickness = 1
            status = f"FAIL ({area_percentage:.1f}%)"

        # Draw the rectangle and label
        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, thickness)
        text = f"{label} | {status}"
        cv2.putText(img_drawn, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 6. Save the annotated debug image
    debug_save_path = os.path.join(output_folder_debug, f"annotated_test_{i}.jpg")
    cv2.imwrite(debug_save_path, img_drawn)

    # Print a status update to the console
    if best_box:
        print(f"Image {i}: Found passing object. Saved Debug & Crop.")
    else:
        print(f"Image {i}: No passing object. Saved Debug only.")