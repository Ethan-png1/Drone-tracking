import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from pathlib import Path


starting_folder = r"C:\Users\jegma\OneDrive\Desktop\SD\Ethan-dev\Datasets"

def merge_classes_to_zero(labels_dir):
    """
    Iterates through all .txt files in a directory and changes
    the class index of every line to 0.
    """
    if not os.path.isdir(labels_dir):
        print(f"Directory not found: {labels_dir}. Skipping.")
        return

    print(f"--- Processing labels in: {labels_dir} ---")
    file_count = 0
    line_count = 0

    # Loop through every file in the directory
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_dir, filename)
            
            # Read all lines from the file
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            new_lines = []
            modified = False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue  # Skip empty lines
                
                # Check if class is already 0
                if parts[0] != '0':
                    parts[0] = '0'  # Change class index to 0
                    modified = True
                    line_count += 1
                
                new_lines.append(" ".join(parts))

            # Only rewrite the file if changes were actually made
            if modified:
                try:
                    with open(file_path, 'w') as f:
                        f.write("\n".join(new_lines) + "\n") # Add trailing newline
                    file_count += 1
                except Exception as e:
                    print(f"Error writing to {file_path}: {e}")

    print(f"Changed {line_count} lines across {file_count} files in {labels_dir}")

def pick_folder():

    root = tk.Tk()
    root.withdraw()

    folder_path_str = filedialog.askdirectory(
        title="Please select a folder",
        initialdir=starting_folder
    )

    folder_path = Path(folder_path_str)
    
    return folder_path

def main():
    dataset = pick_folder()

    train_labels = os.path.join(dataset, "train", "labels")
    test_labels = os.path.join(dataset, "test", "labels")
    val_labels = os.path.join(dataset, "val", "labels")    

    merge_classes_to_zero(train_labels)
    merge_classes_to_zero(test_labels)
    merge_classes_to_zero(val_labels)

    print("--- All label classes merged to 0 ---")

if __name__=='__main__':
    main()