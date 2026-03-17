import os
from pathlib import Path
import shutil

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
starting_folder = r"d:\SD\Ethan-dev\Datasets"
mergeroom_folder = r"d:\SD\Ethan-dev\Datasets\Merge_room"

def duplicate_all(image_folder, dest_folder, count, class_map):
    
    # 1. DEFINE SOURCE PATHS
    parent_dir = os.path.dirname(image_folder)
    source_labels_dir = os.path.join(parent_dir, "labels")

    # 2. DEFINE DESTINATION PATHS
    dest_image_folder = os.path.join(dest_folder, "images")
    dest_labels_dir = os.path.join(dest_folder, "labels")

    # 3. CREATE DESTINATION DIRS (ALWAYS)
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True) # Must always be created

    # 4. CHECK SOURCE LABELS DIR
    if not os.path.isdir(source_labels_dir):
        print(f"Warning: Source 'labels' folder not found at {source_labels_dir}. Will create empty labels.")
        source_labels_dir = None # Set to None, but proceed

    print(f"--- Processing: {image_folder} with map: {class_map} ---")

    for filename in os.listdir(image_folder):

        og_path = os.path.join(image_folder, filename)

        if os.path.isfile(og_path) and filename.lower().endswith(IMAGE_EXTENSIONS):

            # --- Define new file names and paths ---
            name, ext = os.path.splitext(filename)
            new_filename = f"Image-{count}{ext}"
            new_filename_no_ext = f"Image-{count}"

            # Define ALL destination paths unconditionally
            new_image_path = os.path.join(dest_image_folder, new_filename)
            new_label_path = os.path.join(dest_labels_dir, new_filename_no_ext + ".txt")

            # 1. Copy the image
            shutil.copy2(og_path, new_image_path)
            
            # --- Process label file ---
            new_lines = []
            original_label_found = False # Flag to track if we found a file
            
            if source_labels_dir:
                og_label_path = os.path.join(source_labels_dir, name + ".txt")
                
                if os.path.isfile(og_label_path):
                    original_label_found = True
                    # Original label exists, so filter it
                    with open(og_label_path, 'r') as f_in:
                        for line in f_in:
                            parts = line.strip().split()
                            if not parts: continue
                            
                            old_class = parts[0]
                            if old_class in class_map:
                                new_class = class_map[old_class]
                                new_line = f"{new_class} {' '.join(parts[1:])}"
                                new_lines.append(new_line)
            
            # 2. ALWAYS write the new label file (even if empty)
            with open(new_label_path, 'w') as f_out:
                f_out.write("\n".join(new_lines))
                if new_lines:
                    f_out.write("\n")

            # 3. Print a more accurate warning
            if not original_label_found and source_labels_dir:
                # Only print if the source dir existed but the file was missing
                print(f"Warning: Label file not found for {filename}")
            
            count += 1
            
    print(f"Copied and remapped {count-1} images/labels to {dest_folder}.")
    return count
         
def main():
    datasets = mergeroom_folder
    
    if not os.path.exists(datasets):
        print(f"Error: Could not find Merge_room at {datasets}")
        return
        
    total_count = 1

    MAP_DATASET1 = { '0': '0' } 
    
    new_dir_name = input("What is the name of the new combined dataset: ")

    new_dataset_path = os.path.join(starting_folder, new_dir_name)

    print(f"New dataset will be created at: {new_dataset_path}")
    
    # Just create the base folders without train/val/test
    os.makedirs(os.path.join(new_dataset_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, "labels"), exist_ok=True)

    if datasets:
        for root, dirs, files in os.walk(datasets, topdown=True):
            current_map = MAP_DATASET1
            
            if current_map is None:
                continue

            current_folder = os.path.basename(root)
            
            if current_folder.lower() == "images":
                print(f"Processing folder: {root}")
                # duplicate_all expects to write to dest_folder/images and dest_folder/labels
                # so pass new_dataset_path directly to it
                total_count = duplicate_all(root, new_dataset_path, total_count, current_map)

    print("--- Processing Complete ---")
    print(f"Total merged images: {total_count - 1}") 
    

if __name__=='__main__':
    main()
