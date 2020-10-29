import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

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