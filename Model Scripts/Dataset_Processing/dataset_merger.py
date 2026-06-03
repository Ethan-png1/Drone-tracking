import os
from pathlib import Path
from tkinter import filedialog
from tkinter import *
import tkinter as tk
import shutil

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

_ROOT = Path(__file__).resolve().parents[2]
starting_folder = str(_ROOT / "Datasets")

def pick_folder():

    root = tk.Tk()
    root.withdraw()

    folder_path_str = filedialog.askdirectory(
        title="Please select a folder",
        initialdir=starting_folder
    )

    folder_path = Path(folder_path_str)
    
    return folder_path

def duplicate_all(image_folder, dest_folder, count, class_map):
    
    # 1. DEFINE SOURCE PATHS
    parent_dir = os.path.dirname(image_folder)
    source_labels_dir = os.path.join(parent_dir, "labels")

    # 2. DEFINE DESTINATION PATHS
    dest_image_folder = os.path.join(dest_folder, "images")
    dest_labels_dir = os.path.join(dest_folder, "labels")

    # 3. CREATE DESTINATION DIRS (ALWAYS) <-- FIX 1
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
            new_label_path = os.path.join(dest_labels_dir, new_filename_no_ext + ".txt") # <-- FIX 2

            # 1. Copy the image
            shutil.copy2(og_path, new_image_path)
            
            # --- Process label file ---
            new_lines = []
            original_label_found = False # Flag to track if we found a file
            
            if source_labels_dir:
                og_label_path = os.path.join(source_labels_dir, name + ".txt") # <-- Cleaned up
                
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

            # 3. Print a more accurate warning <-- FIX 3 & 4
            if not original_label_found and source_labels_dir:
                # Only print if the source dir existed but the file was missing
                print(f"Warning: Label file not found for {filename}")
            
            count += 1
            
    print(f"Copied and remapped {count-1} images/labels to {dest_folder}.")
    return count
         
def main():
    datasets = pick_folder()
    train_count = 1
    test_count = 1
    val_count = 1

    MAP_DATASET1 = { '0': '0' } 
    
    
    # MAP_DATASET2 = { '0': '0' } 
    
   
    # MAP_DATASET3 = { '0': '0' } 
    
    
    # MAP_DATASET4 = { '0': '0' }

    new_dir_name = input("What is the name of the new dataset: ")

    new_dataset_path = os.path.join(starting_folder,new_dir_name)
    new_train_folder_path = os.path.join(new_dataset_path, "train")
    new_test_folder_path = os.path.join(new_dataset_path, "test")
    new_val_folder_path = os.path.join(new_dataset_path, "val")

   
    #train
    os.makedirs(os.path.join(new_train_folder_path), exist_ok=True)
    #test
    os.makedirs(os.path.join(new_test_folder_path), exist_ok=True)
    #val
    os.makedirs(os.path.join(new_val_folder_path),exist_ok=True)

    print(f"New dataset will be created at: {new_dataset_path}")

    if datasets:
        for root, dirs, files in os.walk(datasets, topdown=True):
            current_map = MAP_DATASET1
            #if "GigaSet" in root: # CHANGE THIS
                # current_map = MAP_DATASET1
            #elif "RealWorld(UAV)" in root: # CHANGE THIS
                # current_map = MAP_DATASET2
            #elif "Drone dataset with birds and whatnot" in root: # CHANGE THIS
                #current_map = MAP_DATASET3
            #elif "dataset" in root: # CHANGE THIS
                #current_map = MAP_DATASET4
            
            if current_map is None:
            # This skips folders that don't match, like the root folder
                continue

            current_folder = os.path.basename(root)
            
            if current_folder == "train":
                print(f"Processing folder: {root}")
                train_count = duplicate_all(os.path.join(root,"images"),new_train_folder_path,train_count, current_map)
            elif current_folder == "test":
                print(f"Processing folder: {root}")
                test_count = duplicate_all(os.path.join(root,"images"),new_test_folder_path,test_count, current_map)
            elif current_folder == "val":
                print(f"Processing folder: {root}")
                val_count = duplicate_all(os.path.join(root,"images"),new_val_folder_path,val_count, current_map)

    print("--- Processing Complete ---")
    print(f"Total Train images: {train_count - 1}") 
    print(f"Total Test images: {test_count - 1}")
    print(f"Total Validation images: {val_count - 1}")
    



if __name__=='__main__':
    main()