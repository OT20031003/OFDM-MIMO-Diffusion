import os
import glob
import random
import shutil
from tqdm import tqdm

# --- Configuration ---
# 1. Specify the path to the extracted COCO images folder
#    (e.g., if you unzipped val2017.zip)
source_dir = './val2017' 

# 2. Destination folder name
dest_dir = 'input_img'
# Remove the previous folder (for safety)
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

# 3. ★Specify the number of images you want here★
num_files_to_copy = 20  # (e.g., copy 50 files)
# --- End of Configuration ---

# Create the destination folder
os.makedirs(dest_dir, exist_ok=True)

# ★Key point★
# Search for .jpg files instead of .png
print(f"Searching for image files in '{source_dir}'...")
all_files = glob.glob(os.path.join(source_dir, '**', '*.jpg'), recursive=True)

# If you want to find multiple extensions (.jpg, .jpeg, .png) at once, use this:
# all_files = []
# for ext in ('*.jpg', '*.jpeg', '*.png'):
#     all_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))


# Main processing
if not all_files:
    print(f"Error: No image files found in '{source_dir}'.")
    print(f"Hint: Check if the 'source_dir' path is correct and if the file extensions (jpg/png) match.")
elif len(all_files) < num_files_to_copy:
    print(f"Error: The total number of image files ({len(all_files)}) is less than the requested number ({num_files_to_copy}).")
    print(f"Please change 'num_files_to_copy' to a value less than or equal to {len(all_files)}.")
else:
    # Select files randomly
    selected_files = random.sample(all_files, num_files_to_copy)
    
    print(f"Copying {num_files_to_copy} images out of {len(all_files)} to '{dest_dir}'...")
    
    # Copy the files
    for file_path in tqdm(selected_files, desc="Copying images"):
        shutil.copy(file_path, dest_dir)
        
    print("Copying complete.")
    print(f"Images have been saved to the '{dest_dir}' folder.")