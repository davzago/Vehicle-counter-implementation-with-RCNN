import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pandas as pd
import json
from pandas import json_normalize

def get_data(number_of_images_for_each_class):
	path = "data/vehicle data/train"

	data = []
	labels = []
	n = number_of_images_for_each_class
	input_dim = (224, 224)

	folders = os.listdir(path)
	folders.sort()
	print("[LOADING IMAGES]")
	for folder in folders:
		files = os.listdir(path + "/" + folder)
		l = n
		if n > len(files):
			l = len(files)
		for f in files[0:l]:
			imagePath = path + "/" + folder + "/" + f
			image = load_img(imagePath, target_size=input_dim)
			image = img_to_array(image)
			image = preprocess_input(image)
			data.append(image)
			labels.append(folder)
	labels = np.unique(labels, return_inverse=True)[1]
	return np.array(data), np.array(labels)

# returns a 2-D array where on each row the first entry corresponds to the imageID
def get_bbox(file_path):
	data = pd.read_csv(file_path, sep=',', usecols=['ImageID', 'XMin', 'XMax', 'YMin', 'YMax'])
	rectangles = []
	for index, row in data.iterrows():
		box = [row['ImageID'], row['XMin'], row['YMin'], row['XMax'], row['YMax']]
		rectangles.append(box)
	return np.array(rectangles)

def get_bbox_json(file_path):
	with open(file_path) as f: 
		d = json.load(f)
	
	df = json_normalize(d)
	rectangles = []
	for _, row in df.iterrows():
		frame_number = row["frameNumber"] - 1 
		objects_json = row["objects"]
		obj_df = json_normalize(objects_json)
		for _, o in obj_df.iterrows():
			box = [frame_number, int(o["bbox.left"]), int(o["bbox.top"]), int(o["bbox.width"]+o["bbox.left"]), int(o["bbox.height"]+o["bbox.top"])]
			rectangles.append(box)
	return rectangles
			
			
		
	
	print(df.head(3))
	"""rectangles = []
	for index, row in data.iterrows():
		box = [row['ImageID'], row['XMin'], row['YMin'], row['XMax'], row['YMax']]
		rectangles.append(box)
	return np.array(rectangles)"""

def compute_iou(box1, box2):
	x1 = max(box1[0], box2[0])
	y1 = max(box1[1], box2[1])
	x2 = min(box1[2], box2[2])
	y2 = min(box1[3], box2[3])

	intersection = max(0, x2-x1 + 1) * max(0, y2-y1 + 1)

	area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
	area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

	return intersection / (area1 + area2 - intersection)

def update_dataset(img_path, ground_truth, dataset_path, max_pos, max_neg, total_pos, total_neg):
	s_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	img = cv2.imread(img_path)
	s_search.setBaseImage(img)
	s_search.switchToSelectiveSearchFast()
	# selective search returns a tuple like this (x, y , w, h)
	r = s_search.process()
	rectangles = []
	for (x, y, w, h) in r:
		x2 = x + w
		y2 = y + h
		rectangles.append([x, y, x2, y2])
	del r
	count_pos = 0
	count_neg = 0
	for (_, x, y, x1, y1) in ground_truth:
		truth_box = [x, y, x1, y1]
		for sel_box in rectangles:
			iou = compute_iou(truth_box, sel_box)
			if count_pos < max_pos and iou > 0.7:
				w = sel_box[2] - sel_box[0]
				h = sel_box[3] - sel_box[1]
				center = (int(sel_box[0]+w / 2), int(sel_box[1]+h / 2))
				c = cv2.getRectSubPix(img, (w, h), center)
				c = cv2.resize(c, (224,224), interpolation=cv2.INTER_CUBIC) #resize to have all images the same size
				cv2.imwrite(dataset_path + '/positive/' + str(total_pos) + '.jpg', c)
				total_pos+=1
				count_pos+=1
			
			fulloverlap = sel_box[0] >= truth_box[0] and sel_box[1] >= truth_box[1] and sel_box[2] <= truth_box[2] and \
				sel_box[3] <= truth_box[3]
			
			if not fulloverlap and iou < 0.1 and count_neg < max_neg:
				w = sel_box[2] - sel_box[0]
				h = sel_box[3] - sel_box[1]
				center = (int(sel_box[0]+w / 2), int(sel_box[1]+h / 2))
				c = cv2.getRectSubPix(img, (w, h), center)
				c = cv2.resize(c, (224,224), interpolation=cv2.INTER_CUBIC)
				cv2.imwrite(dataset_path + '/negative/' + str(total_neg) + '.jpg', c)
				total_neg+=1
				count_neg+=1
	
	return total_pos, total_neg

gt = get_bbox_json("LABELSEXAMPLE.json")
p = 0
n = 0
for frame in range(0,99):
	print(frame)
	# 1. estrarre gt per il frame in esame
	img_gt = list(filter(lambda x: x[0]==frame, gt))
	# 2. update dataset
	p, n = update_dataset("data/frames/"+str(frame)+".jpg", img_gt, "dataset", 20, 15, p, n)
	
