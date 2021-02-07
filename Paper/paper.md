# Vehicle counting with different detection methods

## Abstract

With the objective of learning we implemented a vehicle counting system using non state of the art techniques, we divided this task in two sub-tasks which are detection and tracing. In particular we implement three different methods of detection and one method of tracking then we highlight the strong and weak parts of each method and in the end we discuss possible other implementation and their advantages.

## Introduction

Vehicle counting is the task of counting the number of vehicles passing on a road simply by filming the desired area, without the use of any other sensor.
A system capable of performing this task can be useful to detect automatically high traffic situations, making it easier to inform drivers on the situation ahead.
In our case we decided to perform this task by performing two sub-tasks: detection and tracking.
Detection consists in finding the vehicles in a specific frame of the video, we tried two different methods, the first one uses the difference between 2 subsequent frames in order to find the moving objects meanwhile the second one uses selective search (open cv method) and a CNN on which we performed transfer for the specific task starting from Keras resnet. 
Tracking consists on re-identify the same vehicle between subsequent frames, this will be done by using the pixel distance between the center of the bounding boxes found in the detection phase.

## Related work (EXPAND)

The state of the art systems for vehicle detection or classification use faster RCNN or YOLO networks which basically perform the task of detection but faster by performing detection and classification in one single "look"(look at the yolo net in github). 	

## Detection

### Image Difference

#### General Idea

The intuition behind image differencing is that by subtracting pixel per pixel 2 subsequent images we can obtain only the moving objects which in our case will be vehicles, for this methods to work the requirement is that the camera must be still, in our case we use traffic cameras footage so the requirement is met.
The ideal scenario for this method is having two images one showing just the background and one containing both the background and the object, by subtracting one image to the other (and also some tweaks before and after the subtraction that we will describe later) the result will be the object.

![Background_Subtraction_Tutorial_Scheme](/home/davide/Documenti/Vision/Paper/images/Background_Subtraction_Tutorial_Scheme.png)

 In our case this is impractical to do since we wold need a frame were the road is empty, also we could run into some problems when dealing with different light and or visibility condition hence we will subtract two consecutive frames.

#### Procedure

To perform **Image differencing** first of all we take two consecutive frames and convert them to gray scale, this allows us to avoid to deal with a multichannel image, the result is basically two matrices where each value represents the intensity of a pixel in the image from 0 to 255.
After this simple step we can proceed to compute the **abs difference** between the two images, if the result for a specific element is 0 it means that the same pixel in two consecutive frames had the same intensity and so its contained in the background meanwhile if the result is higher then 0 probably that pixel is part of a moving object.

<img src="/home/davide/Documenti/Vision/Paper/images/gray1.jpg" alt="gray1" style="zoom:80%;" />

Now that we have an image which only contains the moving objects we need a way to draw the corresponding bounding boxes, to do so we will use the open cv function *findcontours* but first we need to transform the grey scale image we have into a binary image.
A binary image is an image consisting in pixels which value is either 0 or 1, this transformation allows the open cv function to check for contours of the objects just by checking where pixels shift from 0 to 1.
In order to transform our image into a binary image we use **thresholding**, this procedure consists in setting a threshold for the intensity of the pixels and put to 0 every pixel below the value and to 1 every pixel above the value.
Setting the right threshold is important, in fact using a low value could highlight unwanted noise meanwhile a high value could result in hiding some parts of an object.

<img src="/home/davide/Documenti/Vision/Paper/images/threshold55.jpg" alt="threshold55" style="zoom:50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/threshold20.jpg" alt="threshold20" style="zoom:50%;" />

Before finally obtaining the contours and consequentially the bounding boxes we dilate the obtained object, by dilating the objects we usually obtain a better object, filling some 0 pixels inside the objects, this can help us to finding better contours.
Practically dilating is done by applying convolution using a kernel containing ones, this operation replaces each pixel value with the maximum pixel value overlapped by the kernel.

Finally we can obtain our contours, using the open cv *findContours* method we obtain the contours in the binary image, the retrieval method decides whether to get all the contours hierarchy or not, the hierarchy  basically is a tree where a more external contour has some son contours internally, in our case we think it's better to keep only the external contours since we only need a bounding box for the vehicles and taking internal contours could highlight some inner parts of in which we are not interested (**KEEP ONLY THE EXTERNAL IS POSSIBLE?**)

Once obtained the contour we use the boundRect method from open cv to obtain our bounding box, this procedure is repeated for each frame of the of the video and since for each step we use two frames the number of frames for the result is n-1 where n is the number of frames of the video.

### Experiment

- video chosen for the detection

- performance in term of detection and time
- bad things (trucks not well detected or multiple detection), when cars are close they can be picked up as a single objects. background picked up
- how to kinda fix and find the balance between this problems

We applied the previously described procedure to a video that can be found on youtube (LINK), we will use the same video for the next experiments as well, the cars are detected quite well and for a video containing 1800 frames in 800x360 resolution the detection is done in 10.5 seconds, this allows this method to be used in real time applications since the video is one minute long.

<img src="/home/davide/Documenti/Vision/Paper/images/well_tracked.jpg" alt="well_tracked" style="zoom:70%;" />

# CORREGGERE QUI

This method doesn't actually detect vehicles but instead it detects moving objects meaning that we can use it if  only vehicles are on the field of view, also it doesn't deal with occlusion hence when a vehicle is behind another one, the detector will find a single blob resulting in a single bounding box, the single detection of two vehicles can happen also when vehicles are very close to each other, in this case the dilate operation can make so the blobs representing the vehicles touch each other resulting in a single blob. By playing with the kernel dimension we see that the bigger it is the more dilated the lines are, having a bigger dilatation is useful to create one single "blob" for one vehicle but by having it too big is risky since we could create one single blob of one or more vehicles if they are close, this is a parameter that could differ between videos, for example in a road with heavy traffic and a far camera the kernel has to be smaller meanwhile if the camera is closer and there is no heavy traffic we could use a bigger kernel.

<img src="/home/davide/Documenti/Vision/Paper/images/no_dilatation.jpg" alt="no_dilatation" style="zoom: 50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/good_dilatation.jpg" alt="good_dilatation" style="zoom: 50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/too_dilated.jpg" alt="too_dilated" style="zoom:50%;" />

Another problem is the detection of trucks, in particular the side of this kind of vehicles is usually wide and monochromatic, this makes so the frame difference in this area results in pixels with intensity close to zero and so when thresholding they are seen as background, because of this the binary image of the trucks shows some separated areas and this leads us to have multiple detection. <img src="/home/davide/Documenti/Vision/Paper/images/multiple_truck.jpg" alt="multiple_truck" style="zoom:50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/truck_blob.jpg" alt="truck_blob" style="zoom:50%;" />

This could be solved by increasing the dimension of the kernel but this would increase the possibility of agglomerating multiple vehicles in a single blob so we found a balance setting the dilate kernel to a 3x3 keeping in mind that this parameter should be changed if using videos with different resolution. 

If the camera moves, for example due to wind, this method will also detect the background as vehicle, this happens because even if an object is stationary it will appear in slightly different position when the camera is shaking, this makes so the difference between frames on specific areas of the background is not close to zero and so it gets detected.
This problem is clear in the initial frames of the video where the camera shakes and there isn't any definitive solution for this problem but in order to mitigate it we decided to only use bounding boxes that have an area higher than sixty pixels, this also helped with the problem of multiple detections for trucks.

<img src="/home/davide/Documenti/Vision/Paper/images/gray_guard.jpg" alt="gray_guard" style="zoom:50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/binary_guard.jpg" alt="binary_guard" style="zoom:50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/guardrail_detection.jpg" alt="guardrail_detection" style="zoom:50%;" />

### RCNN

We also implemented a different method of detection that consists in a region based convolutional neural network that we fine tuned starting from a resnet50 using our own dataset, this is an alternative method of detection that can be chosen when starting the program.
**Our objective is to use selective search on each frame, then give each proposal to the network that will decide weather it's a vehicle or background** 

#### Dataset

Since we are using machine learning for this type of detection we needed a dataset and after trying some dataset we found online that didn't lead to good results we decided to make our own. We recorded a video in Mestre and then used **Labelbox** to draw by hand bounding boxes of the vehicles in some frames, then we used this frames to actually build our training data consisting on a folder containing images of vehicles and a folder containing background using selective search and intersection over union ratio.
Selective Search is a region proposal algorithm from open cv that given an image returns a bunch of bounding boxes containing region that could possibly contain an object, this is done by firs sub-segmenting the image and then by combining this smaller regions into larger ones based on similarity criteria such as color similarity, texture similarity, size similarity and fill similarity.
Intersection over union (IoU) is basically a ratio that describes how much 2 bounding box overlap each other, this metric is useful to understand how much a proposed object overlaps with the ground truth, if the value is 1 it means that the rectangles overlap perfectly meanwhile if the value is 0 the two rectangles aren't even touching.
In order to create our dataset for each frame we ran selective search and then we computed IoU for every bounding box found and the ones with a ratio higher than 0.7 were added to the vehicles folder, the ones with a ratio lower than 0.1 were instead put into the background folder after checking that the rectangle is not fully contained in the ground truth since the IoU ratio could be low if a bounding box is small and contained in a bigger box.
We also added to the background folder crops of the images in which a vehicle is contained but is a small part of the crop, by doing this we wanted to push the network to select smaller bounding boxes that only contain the vehicles.
In order to keep balance between the number of image of background and vehicles we put a different limit to the two types of images that could be produced for each frame, because of this we ended up with about 130 images per class.

### Network

Since we had a very low amount of data we decided to use transfer learning instead of training a full convolutional neural network, transfer learning consists on freezing some layers of a previously trained model and then add some new trainable layers on top of them so they can learn how to turn the old features into prediction on a new dataset.
Following this procedure we took resne50 from keras, which is trained on ImageNet, then we replaced the dense layer with an avarage pooling layer (this layer has benn added because when removing the head of resnet50 the last layer is a convolutional one) followed by three dense layers, two of them use ReLu as activation function meanwhile the last one uses softmax and a total of two neurons because its goal is to distinguish between two classes the input image.
The network during training recives the right label as a one hot vector containing a value 1 on the right class, by using sparse categorical crossentropy as loss function and softmax on the output layer the network outputs a vector containing probability for each class, the class with higher probability is the one that will be selected for the input image.
Since we are working with two classes we could have also used a single output unit with sigmoid as activation function and binary crossentropy as loss function but we chose to not use this method because with soft max we could increase the number of classes (for example adding various type of vehicles) easily just by adding some neurons to the output layer.

### method 

After we built our CNN we could apply our procedure that consists in taking a frame, using selective search in order to obtain some proposal boxes, then we give each of this proposal to the CNN that decides whether it is a vehicle or not, this gave us a bunch of overlapping bounding boxes that contain the vehicles.
In order to obtain a single rectangle per vehicle we had to apply non maxima suppression, our implementation of this method first sorts the boxes based on the probability of them containing a vehicle returned by the network then takes the box with higher probability and suppresses the ones that overlap with it more than a certain IoU ratio, with this we obtained our final vehicle detection for the selected frame.   

### Results 

The results obtained on our test video are good, initially the bounding boxes were sometimes too big around the vehicle and some contained two vehicles but by adding examples of big bounding boxes to the background folder we obtained more narrow bounding boxes, i we wanted to have more precise bounding boxes we should've used a bounding box regressor at the cost of slowing down the process even more.
This method is very slow with respect to the image difference method and definitely can't be used in a real time scenario, in fact in order to compute the results for a single frame the method takes about a minute. Also we noticed that our network isn't very elastic, since we had a very restricted training set the predictions aren't good when the angle of the camera with respect to the road is different from the one in the training set, we realized this when trying to use our RCNN in a video we got from **autostrade italiane** (#SHOW FRAME).
In order to get better results we first of all we should have used a dataset with more examples where the vehicles are depicted from different angles.
In order to obtain a faster detection there are several changes that could have done, first of all we could use a lighter network, but also we could have implemented a fast RCNN or a faster RCNN, in fact our method is so slow because we apply convolution on each region proposal that selective search returns (about 2000 regions per frame) meanwhile the other implementations mentioned above use convolution once on the whole image.

## Tracking

Our objective for the project is to count the passing vehicles, this can't be done performing only detection on each frame, we also need to re-identify the same vehicle in two consecutive frames, to do so decided to use a method based on the distances of the center of the bounding boxes between frames.
The idea is that in two subsequent frames the same vehicle will not move that much thus the center of its bounding box in the first frame will be the closest to the center of its bounding box in the second frame, using this idea we can give an id to each moving object.
Our tracker takes as input the rectangles detected with one of our detection methods and assigns an id to each object detected then updates the center of each objects while going forward with the frames of the video.
Sometimes can happen that we loose track of an object because the detection method fails or the object gets hidden by something else so in order to keep track of the object we added a maximum number of frame where the object can be missing before being deleted from the tracker.    