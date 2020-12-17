from Detector import Detector
from Tracker import Tracker
from RCNN import detect
from RCNN import delete_temp_items
from ToFrame import video_to_frame
import argparse
import cv2
import time
import numpy as np
import os

#array of colors corresponding to each class of vehicle
# verde - 
colors = [(0,255,0), (0,255,255), (0,0,255), (255,0,255), (0,102,205), (0,255,255), (0,128,255), (255,0,127)]  

#array that maps each label to the correct category
categories = ["A truck", "Background", "Bus", "Car", "Motorcycle", "Pickup", "SU truck", "Van"]

parser = argparse.ArgumentParser()
parser.add_argument("--video", help="path of the input video")
parser.add_argument("--input", help="path of the folder where the frames of the video will be stored or are already stored")
parser.add_argument("--output", help="path to the folder where the results will be put")
parser.add_argument("--temp", help="path to the folder where temporary files will be stored")
# Add arguments to choose detection type
#parser.add_argument()
args = parser.parse_args()

input_path = "data/frames"
temp_path = "data/temp"
result_path = "data/results"



if args.input:
    input_path = args.input
    present = os.path.isdir(input_path)
    if not present:
        os.mkdir(input_path)

if args.video:
    video_path = args.video
    present = os.path.isdir(input_path)
    if not present:
        os.mkdir(input_path)
    present = os.path.isfile(video_path)
    if not present:
        raise Exception("there is no video with such path")
    else: 
        video_to_frame(video_path, input_path)

if args.temp:
    temp_path = args.temp
    present = os.path.isdir(temp_path)
    if not present:
        os.mkdir(temp_path)

if args.output:
    result_path = args.output
    present = os.path.isdir(output_path)
    if not present:
        os.mkdir(output_path)


#detector = Detector("data/frames") input_path
tracker = Tracker()
#detector.detect("data/diff") temp_path
# delete_temp_items(temp_path)

"""for i in range(0,len(detector.rects)):
    centers = tracker.update(detector.rects[i])
    img = cv2.imread("data/frames/%d.jpg" %i)
    for j in range(0,len(detector.rects[i])):
        cv2.rectangle(img, detector.rects[i][j][0], detector.rects[i][j][1], (0, 255, 0), 1)
    for c_id, c in centers.items():
        text = "ID{}".format(c_id)
        cv2.putText(img, text, (c[0] - 10, c[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(img, (c[0],c[1]), 4, (0, 255, 0), -1)
        
    cv2.imwrite(result_path + '/%d.jpg' % i, img)"""


vheicles, labels, probs = detect(input_path, temp_path)
for i in range(0,len(vheicles)):
    centers = tracker.update(vheicles[i])
    img = cv2.imread(input_path + "/%d.jpg" %(80+i))
    for j in range(0,len(vheicles[i])):
        color = colors[labels[i][j]]
        cat = categories[labels[i][j]]
        cv2.rectangle(img, (vheicles[i][j][0], vheicles[i][j][1]), (vheicles[i][j][0]+vheicles[i][j][2],vheicles[i][j][1]+vheicles[i][j][3]), color, 1)
        cv2.putText(img, cat, (vheicles[i][j][0], vheicles[i][j][1]-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
    for c_id, c in centers.items():
        text = "ID{}".format(c_id)
        cv2.putText(img, text, (c[0] - 10, c[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1)
        cv2.circle(img, (c[0],c[1]), 4, (0, 255, 0), -1)
        
    cv2.imwrite(result_path + '/%d.jpg' %(i+80), img)


print("the number of vheicles in this video is:", tracker.vheicle_count)
    

    

