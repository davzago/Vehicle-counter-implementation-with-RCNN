# Vehicle counting with different detection methods

## Introduction

Vehicle counting is the task of counting the number of vehicles passing on a road simply by filming the desired area, without the use of any other sensor.
A system that is capable of this can be useful to detect automatically when there is a high traffic situation, making it easier to inform drivers on the situation ahead.
In our case we decided to perform this task by performing two sub-tasks: detection and tracking.
Detection consists in finding the vehicles in a specific frame of the video, we tried two different methods to do this, the first one uses the difference between 2 subsequent frames in order to find the moving objects meanwhile the second one uses selective search (open cv method) and a CNN which we fine tuned for the specific task starting from Keras resnet. 
Tracking consists on re-identify the same vehicle between subsequent frames, this will be done by using the pixel distance between the center of the bounding boxes found in the detection phase.

## Related work 

The state of the art systems for vehicle detection or classification use faster RCNN or YOLO networks which basically perform the task of detection but faster by performing detection and classification in one single "look"(look at the yolo net in github).

## Detection

### Image Difference

#### General Idea

The intuition behind image differencing is that by subtracting pixel per pixel 2 subsequent images we can obtain only the moving objects which in our case will be vehicles, for this methods to work the requirement is that the camera must be still, in our case we use traffic cameras footage so the requirement is met.
The ideal scenario for this method is having two images one showing just the background and one containing both the background and the object, by subtracting one image to the other (and also some tweaks before and after the subtraction that we will describe later) the result will be the object. In our case this is impractical to do since we wold need a frame were the road is empty, also we could run into some problems when dealing with different light and or visibility condition hence we will subtract two consecutive frames. (#IMG OF BACKGROUND SUBTRACTION)

#### Procedure

To perform **Image differencing** first of all we convert the frames in question in gray scale, this allows us to avoid to deal with a multichannel image, the result is basically a matrix where each value represents the intensity of a pixel in the image from 0 to 255.
After this simple step we can proceed to compute the **abs difference** between the two matrices, if the result for a specific element is 0 it means that the same pixel in two consecutive frames had the same intensity and so its contained in the background meanwhile if the result is higher then 0 probably that pixel is part of a moving object. (#EXAMPLE OF GRAYSCALE IMG DIFFERENCE)

Now that we have the image containing only the moving object we need a way to draw a bounding box around it, to do so we will use the open cv function *findcontours* but first we need to transform the gray scale image we have into a binary image.
A binary image is an image consisting in pixel wich value is either 0 or 1, this transformation allows the open cv function to check for contours of the objects just by checking where pixels shift from 0 to 1.
In order to transform our image into a binary image we use **thresholding**, this procedure consists in setting a threshold for the intensity of the pixels and put to 0 every pixel below the value and to 1 every pixel above the value.
Setting the right threshold is important, in fact using a low value could highlight unwanted noise meanwhile a high value could result in hiding some parts of an object so we had to find the right parameter.(#THRESHOLDING IMAGE MAYBE WITH DIFFERENT THRESHOLDS)

Before finally obtaining the contours and consequentially the bounding boxes we dilate the obtained object, by dilating the objects we usually obtain a better object, filling some 0 pixels inside the objects, this can help us to finding better contours and avoid selecting a black part of the object as contour.
Practically dilating is done by applying convolution using a kernel containing ones, this operation replaces each pixel value with the maximum pixel value overlapped by the kernel.
By playing with the kernel dimension we see that the bigger it is the more dilated the lines are, having a bigger dilatation is useful to create one single "blob" for one vehicle but by having it too big the risk is to create one single blob of one or more vehicles if they are close, this is a parameter that could differ between videos, for example in a road with heavy traffic and a far camera the kernel has to be smaller meanwhile if the camera is closer and there is no heavy traffic we could use a bigger kernel.(#IMGAGE OF DILATATION USING DIFFERENT KERNELS MAYBE SHOWING 2 VEHICLES OVERLAPPING)

<img src="/home/davide/Documenti/Vision/Paper/images/no_dilatation.jpg" alt="no_dilatation" style="zoom: 50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/good_dilatation.jpg" alt="good_dilatation" style="zoom: 50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/too_dilated.jpg" alt="too_dilated" style="zoom:50%;" />

Finally we can obtain our contours, using the open cv *findContours* method we obtain the contours in the binary image, the retrieval method decides whether to get all the contours hierarchy or not, the hierarchy  basically is a tree where a more external contour has some son contours internally, in our case we think it's better to keep only the external contours since we only need a bounding box for the vehicles and taking internal contours could highlight some inner parts of in which we are not interested

Once obtained the contour we use the boundRect method from open cv to obtain our bounding box, this procedure is repeated for each frame of the of the video and since for each step we use two frames the number of frames for the result is n-1 where n is the number of frames of the video.

### Experiment

- video chosen for the detection

- performance in term of detection and time
- bad things (trucks not well detected or multiple detection), when cars are close they can be picked up as a single objects. background picked up
- how to kinda fix and find the balance between this problems

We applied the previously described procedure to a video that can be found on youtube (LINK), we will use the same video for the next experiments as well, the cars are detected quite well and for a video containing 1800 frames in 800x360 resolution the detection is done in 10.5 seconds, this allows this method to be used in real time applications since the video is 30 seconds long (#CHECK).

<img src="/home/davide/Documenti/Vision/Paper/images/well_tracked.jpg" alt="well_tracked" style="zoom:70%;" />

There are also some problems with this method, first of all it doesn't deal with occlusion hence when a vehicle is behind another one, the  detector will find a single blob resulting in a single bounding box, the single detection of two vehicles can happen also when vehicles are very close to each other, in this case the dilate operation can make so the blobs representing the vehicles touch each other resulting in a single blob. Another problem is the detection of trucks, in particular the side of this kind of vehicles is usually wide and monochromatic, this makes so the frame difference in this area results in pixels with intensity close to zero and so when thresholding they are seen as background, because of this the binary image of the trucks shows some separated areas and this leads us to have multiple detection. <img src="/home/davide/Documenti/Vision/Paper/images/multiple_truck.jpg" alt="multiple_truck" style="zoom:50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/truck_blob.jpg" alt="truck_blob" style="zoom:50%;" />

This could be solved by increasing the dimension of the kernel but this would increase the possibility of agglomerating multiple vehicles in a single blob so we found a balance setting the dilate kernel to a 3x3 keeping in mind that this parameter should be changed if using videos with different resolution. 

If the camera moves, for example due to wind, this method will also detect the background as vehicle, this happens because even if an object is stationary it will appear in slightly different position when the camera is shaking, this makes so the difference between frames on specific areas of the background is not close to zero and so it gets detected.
This problem is clear in the initial frames of the video where the camera shakes and there isn't any definitive solution for this problem but in order to mitigate it we decided to only use bounding boxes that have an area higher than sixty pixels, this also helped with the problem of multiple detections for trucks.

<img src="/home/davide/Documenti/Vision/Paper/images/gray_guard.jpg" alt="gray_guard" style="zoom:50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/binary_guard.jpg" alt="binary_guard" style="zoom:50%;" /><img src="/home/davide/Documenti/Vision/Paper/images/guardrail_detection.jpg" alt="guardrail_detection" style="zoom:50%;" />

## Tracking
