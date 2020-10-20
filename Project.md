# Veichle counting and classification

## Methods 

- **Frame Differencing:** consists in using 2 neighbor frames 2 determine the moving objects, we use the difference of the 2 frames to obtain contours for the objects.
  This method is useful because its unsupervised unlike deep learning methods which needs som ground truth both when building a model from scratch or when fine applying fine tuning on an already existing one (https://www.analyticsvidhya.com/blog/2020/04/vehicle-detection-opencv-python/) ( we could use this to obtain some bounding box to then classify the veichle)
- **Region based CNN:** this method is the base of object detection, it is based on region of interest methods which first select some areas of the image and then apply conv NN to select the objects in them, also using this method we can classify and get a better(more tight) bounding box over the object using regression (https://www.analyticsvidhya.com/blog/2018/10/a-step-by-step-introduction-to-the-basic-object-detection-algorithms-part-1/?utm_source=blog&utm_medium=vehicle-detection-opencv-python, IMPLEMENTATION: https://www.analyticsvidhya.com/blog/2018/11/implementation-faster-r-cnn-python-object-detection/)
- YOLO and implementation: (https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/)
- **SlimYOLO:** basically taking a Yolo NN, training it and prune the less important connection based on the L1 norm on the weights, after this we fine tune the net and evaluate performance, we can decide to repeat this steps to obtain a slimmer network, this net usually performs worse than RCNN but we gain evaluation speed, this is the tradeoff (https://www.analyticsvidhya.com/blog/2019/08/introduction-slimyolov3-real-time-object-detection/?utm_source=blog&utm_medium=vehicle-detection-opencv-python)
- **Detection and teacing:** https://medium.com/hal24k-techblog/how-to-track-objects-in-the-real-world-with-tensorflow-sort-and-opencv-a64d9564ccb1
- **Simple object traccking with openCV:** https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

## ToDo

implement a simple detector using frame differencing in order to obtain bounding boxes for the veichles, forward the image with bounding boxes to a tracker wich will assign the same id to the same vaichle in different frames by calculating the centroid of the rectangle and finding the closest centroid of the next frame, we need to set a number of frame in which the id will be forgotten 

After this we will substitute the simple frame differencing detectro with a RCNN or a YOLO and compare performance

## Now Using

https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

https://www.analyticsvidhya.com/blog/2020/04/vehicle-detection-opencv-python/

## useful stuff

### Dilating kernel

by playing with the dilate kernel i see that the bigger it is the more dilated the lines are, having a bigger dilatation is useful to create one single "blob" for one veichle but by having it too big the risk is to create one single blob of one or more veichles if they are close, this is a parameter that could differ between videos, i.e. in a road with heavy traffic and a far camera the kernel has to be smaller meanwhile if the camera is closer and there is no heavy traffic we could use a bigger kernel

### Contours retrival method

cv2.findContours() finds the contours in a binary image, the retrival method decides wheter to get all the contours heirarchy (cv2.RETR_TREE) which baiscally is a tree where a more external contour has some son contours internally, in our case i think it's better to keep only the external contours since we only need a bounding box for the veichles 

## Obtain a more clear difference

- The threshold used to transform the grayscale image and the size og the dilate kernel are two parameters wich are important in order to have a good detection, also they have to be tuned together, a high threshold leads us to not include some parts of the veichle risking multiple detection for trucks, dilatation can compensate for that but if we choose a kernel that is too big we risk to agglomerate more veichles 
- to emiminte the part of the background that goes trough the difference we could also not search for veichles in a specific area but this is not possible in the lanes otherwise we would not find veichles
- Also to obtain a better difference i subtract frames that are 10 frames apart

### frame differencing problems

The frame differencing method seems useful but it has very clear limits:

- if 2 veichles are overlapping it will detect one single veichle since the contour will contain both the veichles 
- When detecting a truck this method will find more veichles this happens because a truck's side is plain, this leads the difference between frame to have disconnected parts even it's a single truck
- If the camera is shaking frame differencing will highlight some part of the background which could be detected as veichles, if the camera shakes very hard this method is impossible to use 
- Shadows will get counted as part of the veichle 

this problem makes so this method is not usable in high traffic areas where we can have multiple veichles overlapping, to solve this problem we could use blob(?) detector wich compares pixels in order to detect the objects

## Links for data and models 

- https://www.pyimagesearch.com/2020/07/06/region-proposal-object-detection-with-opencv-keras-and-tensorflow/https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models 
- https://storage.googleapis.com/openimages/web/visualizer/index.html?set=valtest&type=detection&c=%2Fm%2F07yv9
- https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55
- https://www.pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/
- https://www.pyimagesearch.com/2020/07/06/region-proposal-object-detection-with-opencv-keras-and-tensorflow/
- https://www.pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/

## possible dataset

http://podoce.dinf.usherbrooke.ca/challenge/dataset/