import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pandas as pd

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

get_bbox("/home/davide/Scaricati/oidv6-train-annotations-bbox.csv")
	