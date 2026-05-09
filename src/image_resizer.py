import os
from PIL import Image


def pad_and_resize(folder_path, target_size=224):
    # Find all images in the IRR folder and subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)

                try:
                    # Open the image
                    img = Image.open(img_path).convert('RGB')
                    w, h = img.size

                    # If it's already exactly 224x224, skip it
                    if w == target_size and h == target_size:
                        continue

                    # 1. Find the longest side to make a square
                    max_dim = max(w, h)

                    # 2. Create a new black square image
                    square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))

                    # 3. Paste the original 16:9 image exactly in the center
                    paste_x = (max_dim - w) // 2
                    paste_y = (max_dim - h) // 2
                    square_img.paste(img, (paste_x, paste_y))

                    # 4. Resize the padded square down to 224x224
                    # (Using high-quality Lanczos resampling)
                    final_img = square_img.resize((target_size, target_size), Image.LANCZOS)

                    # Overwrite the original image
                    final_img.save(img_path)
                    print(f"Fixed: {img_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


# Run the function on your IRR folder
print("Looking for images...")
target_dir = r'../data/IRR/raw'

if not os.path.exists(target_dir):
    print(f"ERROR: Could not find the folder at {target_dir}")
else:
    print("Folder found! Starting image conversion...")
    pad_and_resize(target_dir)
    print("All images are now 224x224 squares!")