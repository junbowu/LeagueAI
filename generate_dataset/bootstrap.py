import numpy as np
from PIL import Image
from os import listdir
import random
####### TODO ##############
#- crop final image to 1200x800 or maybe squared?
#- cluster the objects to force more overlay, especially for minions
####### Params ############
# Directory in which the masked object images are located
masked_images_dir = "/home/oli/Workspace/LeagueAI/LeagueAI_dataset/masked_objects"
# Directory in which the masked minion images are located
masked_minions = "/home/oli/Workspace/LeagueAI/LeagueAI_dataset/masked_minions"
# Directory in which the map backgrounds are located
map_imags_dir = "/home/oli/Workspace/LeagueAI/LeagueAI_dataset/map"
# Directory in which the tower images are located
tower_dir = "/home/oli/Workspace/LeagueAI/LeagueAI_dataset/masked_towers"
# Directory in which the dataset will be stored (creates jpegs and labels subdirectory there)
output_dir = "/home/oli/Workspace/LeagueAI/LeagueAI_dataset/LeagueAIDataset"
# Which class the current object has
obj_class = 0
# Prints a box around the placed object in red (for debug purposes)
print_box = True
# Size of the datasets the program should generate
dataset_size = 1
# How many characters should be added minimum/maximum to each sample
characters_min = 0
characters_max = 4
assert (characters_min < characters_max), "Error, minions_max needs to be larger than minions_min!"
# How many minons should be added minimum/maximum to each sample
minions_min = 0
minions_max = 8
assert (minions_min < minions_max), "Error, minions_max needs to be larger than minions_min!"
# The scale factor of how much a champion image needs to be scaled to have a realistic size
scale = 0.45
########### Helper functions ###################
"""
This function places a masked image with a given path onto a map fragment
"""
def add_object(path, cur_image_path):
    # Set up the map data
    map_image = Image.open(cur_image_path)
    map_image = map_image.convert("RGBA")
    map_data = map_image.getdata()
    w, h = map_image.size
    print("Adding object: ", path)
    # Read the image file of the current object to add
    obj = Image.open(path)
    obj_w, obj_h = obj.size
    # Rescale the image based on the scale factor
    size = int(obj_w*scale), int(obj_h*scale)
    obj.thumbnail(size)
    # Convert to RGBA to add an alpha channel
    obj = obj.convert("RGBA")
    obj_w, obj_h = obj.size
    # Compute a random position to place the object within the bounds of the map frame
    obj_pos_center = (random.randint(0, w), random.randint(0, h))
    print("Placing at : {}|{}".format(obj_pos_center[0], obj_pos_center[1]))
    # Extract the image data
    obj_data = obj.getdata()
    out_data = []
    # Compute the lowest x and y value that the image has in the gloabl map fragment
    obj_x_offset = obj_pos_center[0] - int(obj_w / 2)
    obj_y_offset = obj_pos_center[1] - int(obj_h / 2)

    # Place the images
    for y in range(0, h):
        for x in range(0, w):
            pixel = (0, 0, 0, 0)
            # Compute the pixel index in the map fragment
            map_index = x + w * y
            # If we want to print the box around the object, set the pixel to red
            if print_box is True and y == obj_pos_center[1] - int(obj_h / 2) and \
            x > obj_pos_center[0] - int(obj_w / 2) and x < obj_pos_center[0] + int(obj_w / 2):
                pixel = (255, 0 ,0, 255)
            elif print_box is True and y == obj_pos_center[1]+int(obj_h/2) and \
            x > obj_pos_center[0]-int(obj_w/2) and x < obj_pos_center[0]+int(obj_w/2):
                pixel = (255,0,0,255)
            elif print_box is True and x == obj_pos_center[0]-int(obj_w/2) and \
            y > obj_pos_center[1]-int(obj_h/2) and y < obj_pos_center[1]+int(obj_h/2):
                pixel = (255,0,0,255)
            elif print_box is True and x == obj_pos_center[0]+int(obj_w/2) and \
            y > obj_pos_center[1]-int(obj_h/2) and y < obj_pos_center[1]+int(obj_h/2):
                pixel = (255,0,0,255)
            else:
                # Replace the old input image pixels with the object to add pixels
                if x >= obj_pos_center[0] - int(obj_w/2) and x <= obj_pos_center[0] + int(obj_w/2) \
                and y >= obj_pos_center[1] - int(obj_h/2) and y <= obj_pos_center[1] + int(obj_h/2):
                    obj_x = x - obj_pos_center[0] - int(obj_w / 2) + 1
                    obj_y = y - obj_pos_center[1] - int(obj_h / 2) + 1
                    object_index = (obj_x + obj_w * obj_y)
                    # Check the alpha channel of the object to add
                    # If it is not 255, the image is seethrough
                    # Then use the original images pixel value
                    # Else use the object to adds pixel value
                    if obj_data[object_index][3] is 255:
                        pixel = (obj_data[object_index][0],  obj_data[object_index][1], obj_data[object_index][2], 255)
                    elif obj_data[object_index][3] is 0:
                        pixel = (map_data[map_index][0], map_data[map_index][1], map_data[map_index][2], 255)
                else:
                    pixel = (map_data[map_index][0], map_data[map_index][1], map_data[map_index][2], map_data[map_index][3])
            out_data.append(pixel)
    # Save the image
    map_image.putdata(out_data)
    map_image = map_image.convert("RGB")
    map_image.save(output_dir+"/jpegs/"+filename+".jpg", "JPEG")


########### Main function ######################
obj_dirs = sorted(listdir(masked_images_dir))
maps = sorted(listdir(map_imags_dir))

for dataset in range(0, dataset_size):
    print("Dataset: ", dataset)
    filename = str(dataset)
    # Randomly select a map background
    mp_fnam = map_imags_dir+"/"+random.choice(maps)
    print("Using map fragment: ", mp_fnam)

    # Randomly select a set of characters to add to the image
    characters = []
    for i in range(0, random.randint(characters_min, characters_max)):
        # Select a random object that we want to add
        temp_obj_folder = random.choice(obj_dirs)
        temp_obj_path = masked_images_dir+"/"+temp_obj_folder
        # Select a random masked image of that object
        characters.append(masked_images_dir+"/"+temp_obj_folder+"/"+random.choice(sorted(listdir(temp_obj_path)))) 
    print("Adding {} champions!".format(len(characters)))

    # Randomly add 0-12 minions to the image
    minions = []
    for i in range(minions_min, minions_max):
        minions.append(masked_minions+"/"+random.choice(sorted(listdir(masked_minions))))
    print("Adding {} minions!".format(len(minions)))

    # Randomly select one tower image
    towers = []
    for i in range(0, 1):
        towers.append(tower_dir+"/"+random.choice(sorted(listdir(tower_dir))))
    print("Adding 1 tower!")


    # Now figure out the order in which we want to add the objects (So that sometimes objects will overlap)
    objects_to_add = characters+minions+towers
    random.shuffle(objects_to_add)
    # Read in the current map background as image
    map_image = Image.open(mp_fnam)
    map_image.save(output_dir+"/jpegs/"+filename+".jpg", "JPEG")
    cur_image_path = output_dir+"/jpegs/"+filename+".jpg"
    # Iterate through all objects in the order we want them to be added and add them to the backgroundl
    # Note this function also saves the image already
    for i in range(0, len(objects_to_add)):
        add_object(objects_to_add.pop(), cur_image_path)

