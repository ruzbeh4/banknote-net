import os
import glob

# Update these paths to match exactly where you extracted the dataset
label_folders = [
    "./data/yolo/train/labels",
    "./data/yolo/valid/labels",
    "./data/yolo/test/labels"  # If a test folder exists
]

files_modified = 0

for folder in label_folders:
    if not os.path.exists(folder):
        continue

    # Find every .txt file in the folder
    txt_files = glob.glob(os.path.join(folder, "*.txt"))

    for file_path in txt_files:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                # OVERRIDE: Force the first item (Class ID) to be '0'
                parts[0] = '0'
                new_lines.append(" ".join(parts) + "\n")

        # Save the file back
        with open(file_path, 'w') as file:
            file.writelines(new_lines)

        files_modified += 1

print(f"Dataset successfully cleaned! Modified {files_modified} annotation files.")