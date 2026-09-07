# use this script when you want to squash images after crop to accepted 224*224 banknote-net input
import argparse
import os
from PIL import Image, ImageOps


def squash_and_resize(input_folder, output_folder, target_size=224):
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, input_folder)
                target_subfolder = os.path.join(output_folder, rel_dir)
                os.makedirs(target_subfolder, exist_ok=True)
                dest_path = os.path.join(target_subfolder, file)

                try:
                    # 2. Open the image
                    img = Image.open(img_path)

                    # 3. Apply the EXIF rotation permanently to the pixels
                    img = ImageOps.exif_transpose(img)

                    # 4. Now convert to RGB
                    img = img.convert('RGB')

                    w, h = img.size

                    if w == target_size and h == target_size:
                        img.save(dest_path)
                        continue

                    final_img = img.resize((target_size, target_size), Image.LANCZOS)
                    final_img.save(dest_path)
                    print(f"Saved: {dest_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Resize cropped images to 224x224.")
    parser.add_argument("--input_dir", default=r'./data/crop_test/output_cropped')
    parser.add_argument("--output_dir", default=r'./data/IRR/processed-filtered-by-yolo/test/None')
    return parser.parse_args()


args = parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

print(f"Looking for images in: {input_dir}")

if not os.path.exists(input_dir):
    print(f"ERROR: Could not find the folder at {input_dir}")
else:
    print(f"Starting conversion...\nOriginals kept in: {input_dir}\nSaving squashed copies to: {output_dir}")
    squash_and_resize(input_dir, output_dir)
    print("\nSuccess! All images are squashed to 224x224 and saved in the new folder structure.")