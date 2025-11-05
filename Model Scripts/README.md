# -Dataset merger

This script will merge datasets into one and leave the old dataset intact. 
The first step to using this script is going in and changing this line of code

starting_folder = r"C:\Users\jegma\OneDrive\Desktop\SD\Ethan-dev\Datasets"

to whatever path your Datasets branch is on.

Next i will outline what your folder setup should look like.

Datasets - This will be the root of everything, where all the magic will happen

Datasets\Merge_room - This is where you store the datasets you want to merge. Does not need to be named the same. 

Datasets\Merge_room\ex_dataset - This is an example of one of the dataset that you will merge. It should contain a train folder, test folder, and val folder all in YOLO format (.txt file that links to a image file). The folder names are case senstive.

Datasets\Merge_room\ex_dataset\train (test or val) - Each of these folders must have a images and labels folder. Also case senstive.

Now when mergering datasets the labels.txt will have a line like this

0 0.450 0.700 0.100 0.050

The 0 is the class id and is the only roadblock. The datasets you want to merge may label drones as any class id number (0,1,2,.ie). There is a auto re_class_id process in the script. 

To setup first decided on a class id for the drones. ex. 0

Then go to this section of code

MAP_DATASET1 = { '0': '0' } 
    
MAP_DATASET2 = { '1': '0' } 
    

MAP_DATASET3 = { '1': '0' } 
    

MAP_DATASET4 = { '0': '0' }

# example foramt{'x': 'y'}

This lets you tell the program how you want each dataset relabeled. The left x is what you are looking to rename in the dataset labels, the right y what you change it too. The right y should be the same for all of them.

Finally this code is what needs to be changed

if "Anti drone hybrid image dataset" in root: # CHANGE THIS
                current_map = MAP_DATASET1
            elif "updated dataset" in root: # CHANGE THIS
                current_map = MAP_DATASET2
            elif "Drone dataset with birds and whatnot" in root: # CHANGE THIS
                current_map = MAP_DATASET3
            elif "dataset" in root: # CHANGE THIS
                current_map = MAP_DATASET4

Change the names in the if statements to the names of the datasets being merged in the merge_room and then put the cooresponding MAP to the dataset. 


When you run the code it should open a window and ask you to select a folder, you should select you merge room folder and then it will ask you to name the new dataset. This new dataset will be made in the Datasets folder.

Do not worry if a message apperas that a file was missing, it may happen and just means the image file didn't have a label.txt file to go with it and the code will create a blank .txt file for it.