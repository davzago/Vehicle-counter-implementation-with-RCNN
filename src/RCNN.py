import numpy as np
import cv2
import time
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os


def delete_temp_items(temp_path):
    files = os.listdir(temp_path)
    for f in files:
        os.remove(temp_path + "/" + f) 

def get_boxes(rectangles):
    if len(rectangles)==0:
        return rectangles, -1, -1

    for k in range(0,len(rectangles)):
        rectangles[k,2] = rectangles[k,2] - rectangles[k,0]
        rectangles[k,3] = rectangles[k,3] - rectangles[k,1]
    return rectangles[:,:4].astype("int"), rectangles[:,4].astype("int"), rectangles[:,5]


def nms(rectangles, overlap_threshold):
    if len(rectangles)==0:
        return rectangles, -1, -1
    x1 = rectangles[:,0]
    y1 = rectangles[:,1]
    x2 = rectangles[:,2]
    y2 = rectangles[:,3]
    labels = rectangles[:,4]  
    prob = rectangles[:,5]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indexes = np.argsort(prob)

    pick = []

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[indexes[:last]])
        yy1 = np.maximum(y1[i], y1[indexes[:last]])
        xx2 = np.minimum(x2[i], x2[indexes[:last]])
        yy2 = np.minimum(y2[i], y2[indexes[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[indexes[:last]] #area[i]

        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    for k in pick:
        rectangles[k,2] = rectangles[k,2] - rectangles[k,0]
        rectangles[k,3] = rectangles[k,3] - rectangles[k,1] 
    return rectangles[pick,:4].astype("int"), rectangles[pick,4].astype("int"), rectangles[pick,5]


def detect(input_path, temp_path):
    s_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    s_search_path = "data/SSearch"
    times = []
    crops = []
    vheicles = []
    probs = []
    final_labels = []
    n_frames = len(os.listdir(input_path))
    for i in range(85,86):
        model = keras.models.load_model('models/bigger_negative_resnet.h5')
        img_original = cv2.imread(input_path + "/%d.jpg" %i)
        maxX = len(img_original[0])
        maxY = len(img_original)
        img = cv2.resize(img_original, (600,360), interpolation=cv2.INTER_CUBIC)
        max_height, max_width, _ = img.shape
        s_search.setBaseImage(img)
        s_search.switchToSelectiveSearchFast()
        #s_search.switchToSelectiveSearchQuality()
        print("starting selective search on image %d" %i)
        rectangles = s_search.process()
        crop = []
        count = 0
        v = []
        for (x, y , w, h) in rectangles:
            center = (int(x+w/2), int(y+h/2))
            c = cv2.getRectSubPix(img, (w, h), center)
            #crop.append(c)
            cv2.imwrite(temp_path + "/%d.jpg" %count, c)
            del c
            count += 1
            crop2 = []
        for j in range(0, count):
            temp_img = load_img(temp_path + "/%d.jpg" %j, target_size=(224, 224))
            os.remove(temp_path + "/%d.jpg" %j)
            temp_img = img_to_array(temp_img) 
            crop2.append(preprocess_input(temp_img))
        data = np.array(crop2)
        del crop2
        labels = model.predict(data)
        K.clear_session()
        del data
        del crop
        for k in range(0,len(labels)):
            if labels[k].argmax() == 1: #and labels[k,1] > 0.7:
                #cv2.imwrite(temp_path + "" + '/%d.jpg' % k, crop[k])
                x, y, w, h = rectangles[k]
                idx = labels[k].argmax()
                rect = (int(maxX*x/600), int(maxY*y/360), int(maxX*(x+w)/600), int(maxY*(y+h)/360), idx, labels[k].argmax())
                v.append(rect)
        rect_and_labels = np.array(v)
        #boxes, label, prob = get_boxes(rect_and_labels)
        boxes, label, prob = nms(rect_and_labels, 0.01)  
        #probs.append(prob)
        final_labels.append(label)
        vheicles.append(boxes)
        #rec.append(rectangles)
        #for (x, y, h, w) in rectangles:
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        #cv2.imwrite(s_search_path +'/%d.jpg' % i, img)
    return vheicles, final_labels, probs

def binary_nms(rectangles, overlap_threshold):
    x1 = rectangles[:,0]
    y1 = rectangles[:,1]
    x2 = rectangles[:,2]
    y2 = rectangles[:,3]
    prob = rectangles[:,4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indexes = np.argsort(prob)

    pick = []

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[indexes[:last]])
        yy1 = np.maximum(y1[i], y1[indexes[:last]])
        xx2 = np.minimum(x2[i], x2[indexes[:last]])
        yy2 = np.minimum(y2[i], y2[indexes[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[indexes[:last]] #area[i]

        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    for k in pick:
        rectangles[k,2] = rectangles[k,2] - rectangles[k,0]
        rectangles[k,3] = rectangles[k,3] - rectangles[k,1] 
    return rectangles[pick,:4].astype("int"), rectangles[pick,4]

def binary_detect(input_path, temp_path): #To Do
    s_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    s_search_path = "data/SSearch"
    times = []
    crops = []
    vheicles = []
    probs = []
    final_labels = []
    n_frames = len(os.listdir(input_path))
    for i in range(419,425): #range(0,n_frames)
        model1 = keras.models.load_model('models/binary_resnet.h5')
        #model2 = keras.models.load_model('models/resnet2.h5')
        #model3 = keras.models.load_model('models/resnet3.h5')
        img = cv2.imread(input_path + "/%d.jpg" %i)
        max_height, max_width, _ = img.shape
        s_search.setBaseImage(img)
        s_search.switchToSelectiveSearchFast()
        #s_search.switchToSelectiveSearchQuality()
        print("starting selective search on image %d" %i)
        start = time.time()
        rectangles = s_search.process()
        end = time.time()
        times.append(end-start)
        crop = []
        count = 0
        v = []
        for (x, y , w, h) in rectangles:
            center = (int(x+w/2), int(y+h/2))
            c = cv2.getRectSubPix(img, (w, h), center)
            #crop.append(c)
            cv2.imwrite(temp_path + "/%d.jpg" %count, c)
            count += 1
            crop2 = []
        for j in range(0, count):
            temp_img = load_img(temp_path + "/%d.jpg" %j, target_size=(224, 224))
            os.remove(temp_path + "/%d.jpg" %j)
            temp_img = img_to_array(temp_img) 
            crop2.append(preprocess_input(temp_img))
        data = np.array(crop2)
        del crop2
        labels = model1.predict(data)
        K.clear_session()
        del data
        del crop
        for k in range(0,len(labels)):
            if labels[k]>0.7:
                #cv2.imwrite(temp_path + "" + '/%d.jpg' % k, crop[k])
                x, y, w, h = rectangles[k]
                #idx = labels[k].argmax()
                rect = (x, y, x+w, y+h, labels[k])
                v.append(rect)
        rect_and_labels = np.array(v)
        nms_boxes, label, prob = nms(rect_and_labels, 0.01)  
        probs.append(prob)
        final_labels.append(label)
        vheicles.append(nms_boxes)
        #rec.append(rectangles)
        #for (x, y, h, w) in rectangles:
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        #cv2.imwrite(s_search_path +'/%d.jpg' % i, img)
    return vheicles, final_labels, probs