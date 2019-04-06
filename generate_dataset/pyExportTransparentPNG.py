from PIL import Image
import numpy as np
from os import listdir

################# Parameters ########################
# Input directory of images
input_dir = "/home/oli/Workspace/LeagueAI/raw_data/red_caster_raw/exported_frames"
# Output directory of masked and cropped images
output_dir = "/home/oli/Workspace/LeagueAI/generate_dataset/masked_minions/red_caster"
# Area to pre-crop the images to (min_x, min_y, max_x, max_y), can save runtime for large screenshots with small objects
# Teemo model viewer: (700,300,1240,780)
area = (0,140,1920,940)
# Background color in RGB
# Teemo model viewer pink background (95, 80,170)
background = (62,255,8)
# The threshold for removing the background color
# Teemo model viewer tolerance (25)
tolerance = 120
# This is needed because there is another shade of pink in the background
tolerance_offset_1 = 1.0
tolerance_offset_2 = 0.75 # Greenscreen: 1.5
tolerance_offset_3 = 1.0 # Teemo viewer: 2.5
#####################################################

def get_min_max_x(newData, w, h):
    min_value = 0
    max_value = 0
    for x in range(w-1, 0,-1):
        for y in range(0, h-1):
            data_index = x + w * y
            if newData[data_index][3] is not 0:
                max_value = x
                break
        else:
            continue
        break
    for x in range(0, w-1):
        for y in range(0, h-1):
            data_index = x + w * y
            if newData[data_index][3] is not 0: 
                min_value = x
                break
        else:
            continue
        break
    return min_value, max_value

def get_min_max(newData, w, h):
    min_value = 0
    max_value = 0
    for y in range(h-1, 0,-1):
        for x in range(0, w-1):
            data_index = x + w * y
            if newData[data_index][3] is not 0:
                max_value = y
                break
        else:
            continue
        break
    for y in range(0, h-1):
        for x in range(0, w-1):
            data_index = x + w * y
            if newData[data_index][3] is not 0: 
                min_value = y
                break
        else:
            continue
        break
    return min_value, max_value



# Get list of files in the input directory
files = sorted(listdir(input_dir))

for f in files:
    # Remove the jpg ending
    fname = f.split(".")[0]
    print(fname)
    img = Image.open(input_dir+"/"+f)
    # Add alpha channel
    img = img.convert("RGBA")

    # Pre crop to save runtime
    cropped = img.crop(area)
    datas = cropped.getdata()
    newData = []
    for item in datas:
        if item[0] > background[0] - tolerance_offset_1*tolerance and item[0] < background[0] + tolerance_offset_1*tolerance \
            and item[1] > background[1] - tolerance*tolerance_offset_2 and item[1] < background[1] + tolerance_offset_2*tolerance \
            and item[2] > background[2] - tolerance_offset_3*tolerance and item[2] < background[2] + tolerance_offset_3*tolerance: 
            newData.append((255,255,255,0))
        else:
            newData.append((item[0], item[1], item[2], 255))

    
    # Save new image data
    cropped.putdata(newData)
    w,h = cropped.size
    # Crop image to pixel content
    min_y, max_y = get_min_max(newData, w, h)
    # Trick: rotate the image by 9 degrees and apply the same functions
    temp = cropped.rotate(90)
    temp_w, temp_h = temp.size
    tempData = temp.getdata()
    min_x, max_x = get_min_max_x(newData, w, h)
    # Save output image as png
    cropped = cropped.crop((min_x, min_y, max_x, max_y))
    cropped.save(output_dir+"/"+fname+".png", "PNG")
