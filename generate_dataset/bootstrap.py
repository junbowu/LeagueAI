import numpy as np
from PIL import Image
from os import listdir
import random

####### TODO ##############
# Generate dataset with varying params (also cluster minions tighter maybe a bit)
# Generate dataset with new maske dimages
# Make sure that we are not outside of the image, the bias padding is maybe not enough
# retrain with pretrained weights
# change the classes below
####### Object Classes ####
# 0: Red Tower
# 6: Red Canon
# 7: Red Caster
# 8: Red Melee
# 14: Vayne

####### Params ############
# Print out the status messages and where stuff is placed
verbose = False
# Important the leaf directories have to be called masked_champions, masked_minions, masked_towers or you have to change add_object to write the object classes properly
# Directory in which the masked object images are located
masked_images_dir = "/home/oli/Workspace/LeagueAI/generate_dataset/masked_champions"
# Directory in which the masked minion images are located
masked_minions = "/home/oli/Workspace/LeagueAI/generate_dataset/masked_minions"
# Directory in which the map backgrounds are located
map_imags_dir = "/home/oli/Workspace/LeagueAI/generate_dataset/map"
# Directory in which the tower images are located
tower_dir = "/home/oli/Workspace/LeagueAI/generate_dataset/masked_towers"
# Directory in which the dataset will be stored (creates jpegs and labels subdirectory there)
# Attention if you use darknet to train: the structure has to be exactly as follows:
# - Dataset
# -- images
# --- XYZ0.jpg
# --- XYZ1.jpg
# -- labels
# --- XYZ0.txt
# --- XYZ1.txt
# -- train.txt
# -- test.txt
output_dir = "/home/oli/Workspace/LeagueAI/generate_dataset/Dataset"
# Prints a box around the placed object in red (for debug purposes)
print_box = False
# Size of the datasets the program should generate
dataset_size = 500
# Beginning index for naming output files
start_index = 3000
# How many characters should be added minimum/maximum to each sample
characters_min = 0
characters_max = 2
assert (characters_min < characters_max), "Error, minions_max needs to be larger than minions_min!"
# How many minons should be added minimum/maximum to each sample
minions_min = 1
minions_max = 6
assert (minions_min < minions_max), "Error, minions_max needs to be larger than minions_min!"
# How many towers should be added to each example
towers_min = 0
towers_max = 1
assert (towers_min < towers_max), "Error, towers_max needs to be larger than towers_min!"
# The scale factor of how much a champion image needs to be scaled to have a realistic size
# Also you can set a random factor to create more diverse images
scale_champions = 0.9 # 0.7 good
random_scale_champions = 0.12 # 0.12 is good
scale_minions = 1.3 #1.2 good
random_scale_minions = 0.25
scale_towers = 1.6 # 1.6 good
random_scale_towers = 0.2 # 0.2 good
# Random rotation maximum offset in counter-/clockwise direction
rotate = 5
# Make champions seethrough sometimes to simulate them being in a brush, value in percent chance a champion will be seethrough
seethrough_prob = 7
# Output image size
output_size = (1920,1080)
# Factor how close the objects should be clustered around the bias point
bias_strength = 180 #220 is good
# Resampling method of the object scaling
#sampling_method = Image.BICUBIC
sampling_method = Image.BILINEAR
# Add random noise to pixels
noise = (10,10,10)
# Sometimes randomly add the overlay
overlay_chance = 20
overlay_path = "/home/oli/Workspace/LeagueAI/generate_dataset/overlay.png"
# Dont spawn objects to close to the border of the image, they might disappear
padding = 30
########### Helper functions ###################
"""
This funciton applies random noise to the rgb values of a pixel (R,G,B)
"""
def apply_noise(pixel):
    R = max(0, min(255, pixel[0] + random.randint(-noise[0], noise[0])))
    G = max(0, min(255, pixel[1] + random.randint(-noise[1], noise[1])))
    B = max(0, min(255, pixel[2] + random.randint(-noise[2], noise[2])))
    A = pixel[3]
    return (R, G, B, A)
"""
This function places a masked image with a given path onto a map fragment
Passing -1 to the object class allows you to set objects like the UI that are not affected by rotations bias etc.
"""
def add_object(path, cur_image_path, object_class, bias_point, last):
    # Set up the map data
    map_image = Image.open(cur_image_path)
    map_image = map_image.convert("RGBA")
    # Cut the image to the desired output image size
    map_data = map_image.getdata()
    w, h = map_image.size
    if verbose: 
        print("Adding object: ", path)
    # Read the image file of the current object to add
    obj = Image.open(path)
    if object_class >= 0:
        # Randomly rotate the image, but make the normal orientation most likely using a normal distribution
        obj = obj.rotate(np.random.normal(loc=0.0, scale=rotate), expand=True)
    obj = obj.convert("RGBA")
    obj_w, obj_h = obj.size
    # Rescale the image based on the scale factor
    if object_class == 0: # tower
        scale_factor = random.uniform(scale_towers-random_scale_towers, scale_towers+random_scale_towers)
        size = int(obj_w*scale_factor), int(obj_h*scale_factor)
    elif object_class == 6 or object_class == 7 or object_class == 8: # red canon minion
        scale_factor = random.uniform(scale_minions-random_scale_minions, scale_minions+random_scale_minions)
        size = int(obj_w*scale_factor), int(obj_h*scale_factor)
    elif object_class == 14: # vayne
        scale_factor = random.uniform(scale_champions-random_scale_champions, scale_champions+random_scale_champions)
        size = int(obj_w*scale_factor), int(obj_h*scale_factor)
    else:
        size = int(obj_w), int(obj_h)
    # If the object is a champion make it seethrough sometimes to simulate it being in a brush
    in_brush = False
    if object_class >= 14 and np.random.randint(0,100) > 100-seethrough_prob:
        in_brush = True
    obj = obj.resize(size, resample=sampling_method)
    # Convert to RGBA to add an alpha channel
    obj_w, obj_h = obj.size
    # Compute the position of minions based on the bias point. Normally distribute the mininons around 
    # a central point to create clusters of objects for more realistic screenshot fakes
    # Champions and structures are uniformly distributed
    if object_class >= 14 or (object_class < 6 and object_class >= 0):
        obj_pos_center = (random.randint(0, w), random.randint(0, h))
    else:
        obj_pos_center = (int(np.random.normal(loc=bias_point[0], scale = bias_strength)), int(np.random.normal(loc=bias_point[1], scale=bias_strength)))
    # Catch the -1 object class exception
    if object_class < 0:
        obj_pos_center = (int(w/2), int(h/2))
    if verbose:
        print("Placing at : {}|{}".format(obj_pos_center[0], obj_pos_center[1]))
    # Extract the image data
    obj_data = obj.getdata()
    out_data = []

    last_pixel = 0
    # Place the images
    for y in range(0, h):
        for x in range(0, w):
            pixel = (0, 0, 0, 0)
            # Compute the pixel index in the map fragment
            map_index = x + w * y
            #print("x: ", x, " y: ", y)
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
                    # If it is smaller 150, the pixel is invisible, 255: fully visible, 150: seethrough (brush simulation)
                    # Then use the original images pixel value
                    # Else use the object to adds pixel value
                    if obj_data[object_index][3] is 255:
                        if in_brush and last_pixel % 3==0:
                            # take the map pixel every second time to make the champion seethrough
                            pixel = (map_data[map_index][0], map_data[map_index][1], map_data[map_index][2], map_data[map_index][3])
                            last_pixel += 1
                        else:
                            pixel = (obj_data[object_index][0],  obj_data[object_index][1], obj_data[object_index][2], 255)
                            last_pixel += 1
                    elif obj_data[object_index][3] is 0:
                        pixel = (map_data[map_index][0], map_data[map_index][1], map_data[map_index][2], 255)
                else:
                    pixel = (map_data[map_index][0], map_data[map_index][1], map_data[map_index][2], map_data[map_index][3])
            # Apply random noise to the pixel
            if last:
                pixel = apply_noise(pixel)
            out_data.append(pixel)
    # Save the image
    map_image.putdata(out_data)
    map_image = map_image.convert("RGB")
    map_image.save(output_dir+"/images/"+filename+".jpg", "JPEG")
    # Append the bounding box data to the labels file if the object class is not -1
    if object_class >= 0:
        with open(output_dir+"/labels/"+filename+".txt", "a") as f:
            # Write the position of the object and its bounding box data to the labels file
            # All values are relative to the whole image size
            # Format: class, x_pos, y_pos, width, height
            f.write("" + str(object_class) + " " + str(obj_pos_center[0]/w) + " " + str(obj_pos_center[1]/h) + " " + str(obj_w/w) + " " + str(obj_h/h) + "\n")


########### Main function ######################
obj_dirs = sorted(listdir(masked_images_dir))
maps = sorted(listdir(map_imags_dir))

for dataset in range(0, dataset_size):
    print("Dataset: ", dataset, " / ", dataset_size)
    filename = str(dataset+start_index)
    # Randomly select a map background
    mp_fnam = map_imags_dir+"/"+random.choice(maps)
    if verbose:
        print("Using map fragment: ", mp_fnam)

    # Randomly select a set of characters to add to the image
    # TODO change classification for other champions
    characters = []
    for i in range(0, random.randint(characters_min, characters_max)):
        # Select a random object that we want to add
        temp_obj_folder = random.choice(obj_dirs)
        temp_obj_path = masked_images_dir+"/"+temp_obj_folder
        # Select a random masked image of that object
        if temp_obj_folder == "vayne_masked":
            characters.append([masked_images_dir+"/"+temp_obj_folder+"/"+random.choice(sorted(listdir(temp_obj_path))),14]) 
    if verbose: 
        print("Adding {} champions!".format(len(characters)))

    # Randomly add 0-12 minions to the image
    # TODO change classification for blue and superminions
    minions = []
    for i in range(0, random.randint(minions_min, minions_max)):
        # Select a random subdirectory because the minions are sorted in subdirectories
        minions_dir = random.choice(sorted(listdir(masked_minions)))
        if minions_dir == "red_canon":
            minions.append([masked_minions+"/"+minions_dir+"/"+random.choice(sorted(listdir(masked_minions+"/"+minions_dir))), 6])
        elif minions_dir == "red_caster":
            minions.append([masked_minions+"/"+minions_dir+"/"+random.choice(sorted(listdir(masked_minions+"/"+minions_dir))), 7])
        elif minions_dir == "red_melee":
            minions.append([masked_minions+"/"+minions_dir+"/"+random.choice(sorted(listdir(masked_minions+"/"+minions_dir))), 8])
        else:
            print("Error: Could not find folder: ", minions_dir)
    if verbose: 
        print("Adding {} minions!".format(len(minions)))

    # Randomly select one tower image
    towers = []
    for i in range(0, random.randint(towers_min, towers_max)):
        # Select a random subdirectory because the towers are sortedy in blue/red team folders
        towers_dir = random.choice(sorted(listdir(tower_dir)))
        towers.append([tower_dir+"/"+towers_dir+"/"+random.choice(sorted(listdir(tower_dir+"/"+towers_dir))), 0])
    if verbose:
        print("Adding 1 tower!")


    # Now figure out the order in which we want to add the objects (So that sometimes objects will overlap)
    objects_to_add = characters+minions+towers
    random.shuffle(objects_to_add)
    # Read in the current map background as image
    map_image = Image.open(mp_fnam)
    w, h = map_image.size
    # Make sure the image is 1920x1080 (otherwise the overlay might not fit properly)
    assert (w == 1920 and h == 1080), "Error image has to be 1920x1080" 

    map_image.save(output_dir+"/images/"+filename+".jpg", "JPEG")
    cur_image_path = output_dir+"/images/"+filename+".jpg"
    # Iterate through all objects in the order we want them to be added and add them to the backgroundl
    # Note this function also saves the image already
    # Point around which the objects will be clustered
    bias_point = (random.randint(0+padding, w-padding), random.randint(0+padding, h-padding))
    # Add the overlay, the bias point plays no role here because of the object class (object class -1 is not added to the labels.txt)
    if random.randint(0,100) > 100 - overlay_chance:
        add_object(overlay_path, cur_image_path, -1, bias_point, False)
    for i in range(0, len(objects_to_add)):
        o = objects_to_add.pop()
        if len(objects_to_add) == 0:
            add_object(o[0], cur_image_path, o[1], bias_point, True)
        else:
            add_object(o[0], cur_image_path, o[1], bias_point, False)
    # of no objects were added we still have to create an empty txt file
    with open(output_dir+"/labels/"+filename+".txt", "a") as f:
        f.write("")


    if verbose:
        print("=======================================")

