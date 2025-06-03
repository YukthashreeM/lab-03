#program to resize image
import numpy as np      #we are using alias to nme numpy package
import cv2              #openCV library is being imported in this line as cv2
import matplotlib.pyplot as plt  #importing the submodule of matplotlib called pyplot further i am using alias them as plt'''
#program to capture from the webcam

cam = cv2.VideoCapture(0)
result, image = cam.read() 

if result:
    cv2.imshow("captured picure",image)
    cv2.waitKey()
    cv2.imwrite("image.jpg", image)
else:
    print("no image found")
image = cv2.imread("image.jpg")  #reading the image2.jpg from the disk where the image has same hierarchy as this program and saving it to the veraible named image'''

new = cv2.resize(image, (1200,800)) #resizeing the contents of image veraible
# to 800,1200 pixels and saving the results to new veriable
cv2.imshow('old image', image) #displaying the original image
cv2.waitKey()   # it retains the picture window displayed in previous line until #we close it
cv2.imshow('new resized image',new )#displaying the resized image
cv2.waitKey()
cv2.imwrite("newimage.jpg", new)#writing the contents of resized data present in veriable new to newimage.jpg


blurimage = cv2.blur(image, (50,50))  
cv2.imshow('blurred  image', blurimage)
cv2.waitKey()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Lion', gray_image)
cv2.imwrite(" gray_image.jpg", gray_image)
cv2.waitKey()

#scaling and rotation
print(image.shape)
print( gray_image.shape)
h, w = image.shape[:2]
center = (w / 2, h / 2)
# print(type(center))

mat = cv2.getRotationMatrix2D(center, 90, 1)
rotating = cv2.warpAffine(image, mat, (h, w))
cv2.imshow('rotated', rotating)
cv2.waitKey()

# edge detection
img_blur = cv2.GaussianBlur(image,(3,3), sigmaX=0, sigmaY=0) 
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
# sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
#sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
cv2.imshow('Sobel X Y using Sobel() function', sobelx)
cv2.waitKey()

#segmentation
src = cv2.imread("image4.jpg", cv2.IMREAD_GRAYSCALE);
cv2.imshow("gray scale img",src) 
# Basic threhold example 
th, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY); 
cv2.imshow('grey scale image', dst)
cv2.waitKey()
