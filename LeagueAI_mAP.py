#Copyright 2019 Oliver Struckmeier
#Licensed under the GNU General Public License, version 3.0. See LICENSE for details

# This script us used to compute the detection precision mAP of a model against a test dataset

from LeagueAI_helper import input_output, LeagueAIFramework, detection
import time
import cv2
from os import listdir
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def get_label_box(f):
    with open(f) as label_file:
        objects = label_file.readlines()
    objects = [o.rstrip('\n') for o in objects]
    return objects

def load_classes(names_file):
    f = open(names_file, "r")
    names = f.read().split("\n")[:-1]
    return names

def compute_map(object_box, object_class, all_boxes, w_in, h_in):
    # If its not the same object, skip
    b1_x1, b1_y1, b1_x2, b1_y2 = float(object_box[0]), float(object_box[1]), float(object_box[2]), float(object_box[3])
    #print("b1_x1: {} b1_y1: {} b1_x2: {} b1_y2: {}".format(b1_x1, b1_y1, b1_x2, b1_y2))
    iou = []
    best_match_obect = []
    for i in all_boxes:
        b = i.split(' ')
        if int(object_class) != int(b[0]):
            best_match_obect.append(detection(0,0,0,0,0))
            iou.append(0)
            continue
        x_pos = int(float(b[1])*w_in)
        w = int(float(b[3])*w_in)
        y_pos = int(float(b[2])*h_in)
        h = int(float(b[4])*h_in)
        b2_x1, b2_y1, b2_x2, b2_y2 = x_pos - int(w/2), y_pos - int(h/2), x_pos + int(w/2), y_pos + int(h/2)
        #print("b2_x1: {} b2_y1: {} b2_x2: {} b2_y2: {}".format(b2_x1, b2_y1, b2_x2, b2_y2))
        # Get the coordinates of the intersetcion rectangle
        inter_rect_x1 = max(b1_x1, b2_x1)
        inter_rect_y1 = max(b1_y1, b2_y1)
        inter_rect_x2 = min(b1_x2, b2_x2)
        inter_rect_y2 = min(b1_y2, b2_y2)
        #print("x1: {}, y1: {}, x2: {}, y2: {}".format(inter_rect_x1,inter_rect_y1,inter_rect_x2,inter_rect_y2))
        # Intersection area
        inter_area = max(0, inter_rect_x2 - inter_rect_x1 + 1) * max(0, inter_rect_y2 - inter_rect_y1 + 1)
        #print("inter_area: ", inter_area)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        #print("b1: {}, b2: {}".format(b1_area, b2_area)) 
        iou.append(inter_area / (b1_area + b2_area - inter_area))
        best_match_obect.append(detection(int(b[0]), b2_x1, b2_y1, b2_x2, b2_y2))
    return max(iou), best_match_obect[iou.index(max(iou))]

name_file = "/home/oli/Workspace/LeagueAI/cfg/LeagueAI.names"
names = ['Tower', 'EnemyMinion', 'Vayne']#load_classes(name_file)
image_folder = "test_map/images/"
label_folder = "test_map/labels/"
output_size = (int(1920/2), int(1080/2))
classes_number = 5
mAP_threshold = 0.5

LeagueAI = LeagueAIFramework(config_file="cfg/LeagueAI.cfg", weights="weights/04_16_LeagueAI.weights", names_file="cfg/LeagueAI.names", classes_number = classes_number, resolution=960, threshold = 0.5, cuda=True, draw_boxes=True)

files = sorted(listdir(image_folder))
show_images = True

mAP_avg = [0] * classes_number
mAP_ground_truth = [0] * classes_number
font_size = 2
for i, f in enumerate(files):
    # Get the current frame from the image
    img = Image.open(image_folder+f)
    img = img.convert("RGB")
    R, G, B = img.split()
    img = Image.merge("RGB", [B, G, R])
    w, h = img.size
    frame = np.array(img)

    # Get the list of detected objects and their positions
    objects = LeagueAI.get_objects(frame)
    label_boxes = get_label_box(label_folder+f.split(".")[0]+".txt")
    print("{} Objects out of {} detected!".format(len(objects), len(label_boxes)))
    for o in objects:
        box_object = [o.x_min, o.y_min, o.x_max, o.y_max]
        mAP, best_match_obect = compute_map(box_object, o.object_class,label_boxes, w, h)
        if mAP >= mAP_threshold:
            mAP_avg[int(o.object_class)] += 1
        mAP_ground_truth[int(o.object_class)] += 1
        # Legend
        t_size = cv2.getTextSize("Detection", cv2.FONT_HERSHEY_PLAIN, font_size, 2)[0]
        cv2.rectangle(frame, (0, 0), (t_size[0], t_size[1]), (0,0,255), -1)
        cv2.putText(frame, "Detection", (0, int(t_size[1])), cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], 2)
        t_size = cv2.getTextSize("Label Ground Truth", cv2.FONT_HERSHEY_PLAIN, font_size, 2)[0]
        cv2.rectangle(frame, (0, t_size[1]), (t_size[0], 2*t_size[1]), (255,0,0), -1)
        cv2.putText(frame, "Label Ground Truth", (0, 2*int(t_size[1])), cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], 2)

        # Paint the boxes of label vs detection to visualize how the mAP is computed
        cv2.rectangle(frame, (o.x_min, o.y_min), (o.x_max, o.y_max), (0,0,255), 2)
        cv2.rectangle(frame, (best_match_obect.x_min, best_match_obect.y_min), (best_match_obect.x_max, best_match_obect.y_max), (255,0,0), 2)
        t_size = cv2.getTextSize("iou: {}".format(round(mAP,1)), cv2.FONT_HERSHEY_PLAIN, font_size, 2)[0]
        cv2.rectangle(frame, (o.x - int(t_size[0]/2), o.y - int(t_size[1]/2)), (o.x + int(t_size[0]/2), o.y + int(t_size[1]/2)), (255,0,0),-1)
        cv2.putText(frame, "iou: {}".format(round(mAP,1)), (o.x - int(t_size[0]/2), o.y + int(t_size[1]/2)), cv2.FONT_HERSHEY_PLAIN, font_size, [255, 255, 255], 2)
    if show_images:
        while True:
            # Show the current image
            frame = cv2.resize(frame, output_size)
            cv2.imshow('LeagueAI', frame)
            if (cv2.waitKey(25) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break
    print("true positives: ", mAP_avg)
    print("ground truth: ", mAP_ground_truth)
    temp = [0] * 5
    for t in range(0, len(mAP_avg)):
        if mAP_ground_truth[t] > 0:
            temp[t] = mAP_avg[t]/mAP_ground_truth[t]
        else:
            temp[t] = 0
    print("mAP: ", temp)
    print("{} of {} images done!".format(i+1, len(files)))

temp, mAP_avg, mAP_ground_truth = zip(*sorted(zip(temp, mAP_avg, mAP_ground_truth)))

# Plot the resulting mAPs for each class
index = np.arange(classes_number)

fig = plt.figure()
fig.add_subplot(2,1,1)
p1 = plt.bar(index, mAP_avg, width = 0.35)
p2 = plt.bar(index, [mAP_ground_truth[i]-mAP_avg[i] for i in range(0, classes_number)], width = 0.35, bottom = mAP_avg)
plt.xticks(index, names)
plt.legend((p1[0], p2[0]), ('True positives with mAP threshold = {}'.format(mAP_threshold), 'False positives'))
plt.xlabel("Classes")
plt.ylabel("Total number of class occurence in the test set")

fig.add_subplot(2,1,2)
plt.bar(index, temp, width = 0.35)
plt.xticks(index, names)
plt.xlabel("Classes")
plt.ylabel("mAP")
plt.show()

