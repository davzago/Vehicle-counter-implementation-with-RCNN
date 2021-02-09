import argparse
import numpy as np
from PIL import Image
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument("--video", help="path of the input video")
parser.add_argument("--output", help="path to the folder where the results will be put")
parser.add_argument("--temp", help="path to the folder where temporary files will be stored")
parser.add_argument("--slitpoint", help="fraction of a frame where to place the slitpoint (default is 2 so 1/2)")
parser.add_argument("--dilKernel", help="size of kernel used in dilatation (as a pair, default is (3,5))")
parser.add_argument("--fillKernel", help="size of kernel used in holes filling (as a pair, default is equal to dilatation Kernel)")
parser.add_argument("--noiseKernel", help="size of kernel used in noise removal (as a pair, default is (11,11))")
parser.add_argument("--countourArea", help="minimum size of contours")
args = parser.parse_args()


temp_path = "data/slitMethod/temp"
result_path = "data/slitMethod/output"
slitPoint = 4/3
dilKSize = (1,3)
filKSize = (1,3)
noiseKSize = (11,11)
cArea = 600


if args.video:
    video_path = args.video
    present = os.path.isfile(video_path)
    if not present:
        raise Exception("there is no video with such path")   

if args.temp:
    temp_path = args.temp
    present = os.path.isdir(temp_path)
    if not present:
        os.mkdir(temp_path)

if args.output:
    result_path = args.output
    present = os.path.isdir(result_path)
    if not present:
        os.mkdir(result_path)

if args.slitpoint:
    slitPoint = args.slitPoint

if args.dilKernel:
    dilKSize = eval(args.dilKernel)

if args.fillKernel:
    filKSize = eval(args.fillKernel)

if args.noiseKernel:
    noiseKSize = eval(args.noiseKernel)

if args.countourArea:
    cArea = int(args.countourArea)


print('Opening video %s for slit scanning' % args.video)

# 1. Create slit image

vidcap = cv2.VideoCapture(args.video)

#https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get

frameWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
frameHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))


# This will be the final slit image
slitImg = np.full((frame_count, int(frameWidth), 3), 169, dtype='uint8')

slitHeight = 1 # = height in pixels taken for each frame -- dire che si puo provare a cambiare ma neitest non cambiava motlo
slitPoint = int(frameHeight / slitPoint )  # = y-coordinate of where to take the shot

success,image = vidcap.read()

# see where is the line that will be taken:
image1 = cv2.line(image, (0 , slitPoint), (frameWidth , slitPoint),  (255,0,0) , 5)
cv2.imwrite(temp_path + '/output_primo_frame.png', image1) 

count = 0
while success:
    slitImg[count:count + slitHeight, : , :] = image[slitPoint:slitPoint+slitHeight, : , :]
    success, image = vidcap.read()
    count += 1


output = cv2.cvtColor(slitImg, cv2.COLOR_BGR2GRAY)
output_Image = Image.fromarray(output)
output_Image.save(temp_path + '/temp1.png')




# 2. Object counting: separate background and count resulting blobs

# y-gradients (that is vertical differences in value of neighboring pixels)
# ksize = CV_SCHARR (-1) non buono, troppo sensibile (troppe linee)
# ksize = 1 molto meno sensibile, solo le auto si vedono (anche se poco)
# ksize = 3 si vedono bene sia auto che righe

sobely = cv2.Sobel(output,cv2.CV_8U,0,1,ksize=3)

_, sobely = cv2.threshold(sobely, 50, 255, cv2.THRESH_BINARY) #to black and white


sobely_img = Image.fromarray(sobely).convert("L")
sobely_img.save(temp_path + '/temp2.png')

'''
1. Morphological dilation

'''

dilatationKernel = cv2.getStructuringElement(cv2.MORPH_CROSS,ksize=dilKSize)

dilatated = cv2.dilate(sobely, dilatationKernel, iterations=3)
dilatated_img = Image.fromarray(dilatated).convert("L")
dilatated_img.save(temp_path + '/temp3.png')
print(dilatationKernel)

'''
2. Filling holes
'''
fillingKernel = cv2.getStructuringElement(cv2.MORPH_CROSS,ksize=filKSize)

filled = cv2.morphologyEx(dilatated,cv2.MORPH_CLOSE,fillingKernel)
filled_img = Image.fromarray(filled).convert("L")
filled_img.save(temp_path + '/temp4.png')


'''

3. Remove noise
'''

# opening
openingKernel = np.ones(noiseKSize, dtype=np.uint8)
opened = cv2.morphologyEx(dilatated,cv2.MORPH_OPEN,openingKernel)

opened_img = Image.fromarray(opened).convert("L")
opened_img.save(temp_path + '/temp5.png')

'''
4. Filter blobs by area
'''
# trova i contorni dei blobs e filtra by area

contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
bigContours = []
cars_number = 0


for c in contours:
    if cv2.contourArea(c) > cArea:
        x,y,w,h = cv2.boundingRect(c)
        if w < 2*cArea/3 and w > cArea/10 and h > cArea/60: 
            cars_number +=1
            bigContours.append(c)



print(cars_number)

countured_img = cv2.drawContours(cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR), bigContours, -1, (0,255,0), 3)

countured_img = Image.fromarray(countured_img)


countured_img.save(result_path + '/result.png')


# NB!!!!!! ALCUNI BLOB SI UNISCONO, ALTRI SI SPEZZANO. TUTTAVIA LA DIVERSITà DEI MEZZI 
# FA SI CHE SI COMPENSINO GLI UNI CON GLI ALTRI, DUNQUE IL NUMERO FINALE è PIù O MENO GIUSTO