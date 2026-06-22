import os
from PIL import Image, ImageOps


def bake_exif_rotation_recursive(root_directory):
    # Supported image formats
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # os.walk goes through the root folder and every sub/subsub folder automatically
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.lower().endswith(valid_extensions):
                filepath = os.path.join(dirpath, filename)

                try:
                    # Open the image
                    img = Image.open(filepath)

                    # Extract the EXIF data
                    exif = img.getexif()

                    # 0x0112 is the standard EXIF tag ID for 'Orientation'
                    # We only process and save IF this tag exists
                    if exif and 0x0112 in exif:
                        # Physically rotate and strip the EXIF tag
                        fixed_img = ImageOps.exif_transpose(img)

                        # Overwrite the original
                        fixed_img.save(filepath)
                        print(f"Fixed: {filepath}")
                    else:
                        # Image has no orientation tag, safe to ignore
                        pass

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")


# Replace with your main root dataset folder
dataset_path = ".\data\IRR\processed2-resized\\train"
bake_exif_rotation_recursive(dataset_path)
print("Processing complete!")