import os
import shutil
import random

# --- CONFIGURATION (EDIT THIS) ---
# 1. The path to your current 'train' folder
# (The folder that CONTAINS 'images' and 'labels')
source_train_folder = r'C:\Users\jegma\OneDrive\Desktop\SD\Ethan-dev\Datasets\RealWorld(UAV)\train'

# 2. The path where you want the new 'val' folder to be created
# (Usually usually just parallel to the train folder)
dest_val_folder = r'C:\Users\jegma\OneDrive\Desktop\SD\Ethan-dev\Datasets\RealWorld(UAV)\val'

# 3. How much to move? (0.2 = 20%)
split_ratio = 0.2
# ---------------------------------

def create_val_split():
    # Define the specific subfolders based on your structure
    src_images_dir = os.path.join(source_train_folder, 'images')
    src_labels_dir = os.path.join(source_train_folder, 'labels')

    dst_images_dir = os.path.join(dest_val_folder, 'images')
    dst_labels_dir = os.path.join(dest_val_folder, 'labels')

    # Verify source exists
    if not os.path.exists(src_images_dir):
        print(f"ERROR: Could not find {src_images_dir}")
        return

    # Create destination folders
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    # Get list of images
    print(f"Scanning {src_images_dir}...")
    all_images = [f for f in os.listdir(src_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # Shuffle and select
    random.shuffle(all_images)
    num_to_move = int(len(all_images) * split_ratio)
    images_to_move = all_images[:num_to_move]

    print(f"Moving {num_to_move} images from 'train' to 'val'...")

    count = 0
    for image_file in images_to_move:
        # Construct full paths
        src_img_path = os.path.join(src_images_dir, image_file)
        dst_img_path = os.path.join(dst_images_dir, image_file)

        # Handle Label (Swap extension to .txt)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        src_label_path = os.path.join(src_labels_dir, label_file)
        dst_label_path = os.path.join(dst_labels_dir, label_file)

        try:
            # Move Image
            shutil.move(src_img_path, dst_img_path)

            # Move Label (Only if it exists)
            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dst_label_path)
            
            count += 1
        except Exception as e:
            print(f"Error moving {image_file}: {e}")

    print(f"Done! Successfully moved {count} pairs.")
    print(f"Your new validation set is at: {dest_val_folder}")

if __name__ == "__main__":
    create_val_split()