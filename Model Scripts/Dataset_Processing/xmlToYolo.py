import xml.etree.ElementTree as ET
import os

# --- CONFIGURATION ---
# 1. List ALL your classes exactly as they appear in the XML files
# IMPORTANT: The order here determines the Class ID (0, 1, 2...)
classes = ['drone'] 

# 2. Where are your files?
xml_folder = r'C:\Users\jegma\OneDrive\Desktop\SD\Ethan-dev\Datasets\RealWorld(UAV)\test\Drone_TestSet_XMLs'
output_folder = r'C:\Users\jegma\OneDrive\Desktop\SD\Ethan-dev\Datasets\RealWorld(UAV)\test\labels'
# ---------------------

def convert(size, box):
    """ Converts corner coordinates to center coordinates (normalized) """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(xml_folder):
    if not filename.endswith('.xml'): 
        continue

    # Parse the XML file
    tree = ET.parse(os.path.join(xml_folder, filename))
    root = tree.getroot()
    
    # Get image size to normalize coordinates
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # Prepare the output .txt file
    txt_filename = filename.replace('.xml', '.txt')
    out_file = open(os.path.join(output_folder, txt_filename), 'w')

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        
        # Skip difficult objects if you want (optional)
        if cls not in classes or int(difficult) == 1:
            continue
            
        cls_id = classes.index(cls)
        
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        
        # Convert to YOLO format
        bb = convert((w, h), b)
        
        # Write line: class_id x_center y_center width height
        out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

    out_file.close()

print("Conversion complete!")