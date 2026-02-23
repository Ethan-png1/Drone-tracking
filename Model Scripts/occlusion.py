import os
from PIL import Image
import random


occlusion_folder = r"C:\Users\jegma\OneDrive\Desktop\SD\Ethan-dev\Datasets\occImgs"

def occlude(base_img,oc_array):

    background = Image.open(base_img).convert("RGBA")

    foreground = Image.open(oc_array[random.randrange(len(oc_array))]).convert("RGBA")
    foreground = foreground.resize(background.size,Image.Resampling.LANCZOS)

    alpha = foreground.getchannel('A')
    alpha = alpha.point(lambda p: int(p * (random.uniform(0.3,0.7))))
    foreground.putalpha(alpha)

    background.paste(foreground,(0,0),foreground)

    return background.convert("RGB")

def folder_lister(folder):
    occlusion_array = []

    for filename in os.listdir(folder):
        occlusion_img = os.path.join(folder, filename)
        occlusion_array.append(occlusion_img)
    return occlusion_array



def main():


    for filename in os.listdir(image_folder):
        og_path = os.path.join(image_folder, filename)





           

if __name__=='__main__':
    main()