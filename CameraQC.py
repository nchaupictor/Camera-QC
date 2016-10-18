# Camera QC Script
# 
# Author: N Chau
# Date: 14/10/2016

# Import libraries
# ------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm
import matplotlib.image as mpimg
from scipy.spatial import distance as dist
import os 
import gzip
import numpy as np 
import time
from PIL import Image
import cv2 
import csv
# ------------------------------------------------------------------------
#Function to sort coordinates into    0 ------ 1
                        
#                                     3 ------ 2

def sort_coord(pts):
    xSorted = pts[np.argsort(pts[:,0]),:]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:,:]
    leftMost = leftMost[np.argsort(leftMost[:,1]),:]
    (tl,bl) = leftMost
    D = dist.cdist(tl[np.newaxis],rightMost,"euclidean")[0]
    (br,tr) = rightMost[np.argsort(D)[::-1],:]
    return np.array([tl,tr,br,bl],dtype="float32")

# ------------------------------------------------------------------------

#Open Video Stream 
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

#Set camera parameters 
#vc.set(3,2592) #Width
#vc.set(4,1944) #Height
#vc.set(5,15)
vc.set(10,0) #Brightness
vc.set(11,34) #Contrast
vc.set(12,64) #Saturation
vc.set(15,-13) #Exposure

if vc.isOpened():
    rval,frame = vc.read()
else:
    rval = False

while rval: 
    cv2.imshow("preview",frame)
    rval, frame = vc.read()
    #cv2.namedWindow("capture")
    #cv2.imshow("capture",frame)

    #Loop through 3 times to take 3 images to ensure out of focus isnt an anomaly

    key = cv2.waitKey(20)  
    if key == 27: #Escape Key
        break
    elif key == 13  : #Enter Key 
        rval, frame = vc.read()
        cv2.imwrite("capture1.bmp",frame)
        cv2.namedWindow("capture")
        image = cv2.imread("capture1.bmp")
        cv2.imshow("capture",image)

        pyplot.close()
        #Split channels to RGB
        b,g,r = cv2.split(image)
        #cv2.imshow("r",r)
        #rThresh = cv2.threshold(r,120,255,cv2.THRESH_BINARY)[1]
        #cv2.imshow("g",rThresh)
        
        #cv2.imshow("Red",rROI)
        #cv2.imwrite("rROI.bmp",rROI)
        #imageROI = cv2.rectangle(image.copy(),(300,255),(305,315),(0,255,0),1)
        #cv2.imshow("ROI",imageROI)

        area = 0

        #Hough circle detector 
        imageROI = image.copy()
        #imageG = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #circles = cv2.HoughCircles(imageG,cv2.HOUGH_GRADIENT,1,1,param1=30,param2=11,minRadius=0,maxRadius = 4)
        #print circles
        #circles = np.uint16(np.around(circles))
        #for i in circles[0,:]:
            #cv2.circle(imageG,(i[0],i[1]),i[2],(0,255,0),2)
        #cv2.imshow('circles',imageG)

        #Pre processing
        #Use red channel to find top left coordinate
        #Blur red channel with 5x5 kernel and run adaptive thresholding 
        r = cv2.blur(r,(5,5))
        rThresh = cv2.adaptiveThreshold(r,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,2)
        cv2.imshow("rThresh",rThresh)
        #erosion = cv2.erode(rThresh,np.ones((3,3),np.uint8),iterations = 1)
        edge = cv2.Canny(rThresh,20,150)

        edgeBlur = cv2.Canny(image,20,150)
        cv2.imshow("capture",edgeBlur)

        #Run edge detection and find contour
        (_,contours, _) = cv2.findContours(edge.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours,key = cv2.contourArea, reverse = True)[:10]
        #print contours[1]
        screenCon = None
        #contours = contours[1]

        #Loop through found contours and check if they are square/rectangular 
        for c in contours:
            per = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,0.02 * per, True)

            if len(approx) == 4:
                #screenCon = approx
                screenCon = cv2.minAreaRect(c)
                screenCon = cv2.boxPoints(screenCon)
                screenCon = np.array(screenCon,dtype="int")
                area = cv2.contourArea(screenCon)
                break

            #cv2.drawContours(image,[approx],-1,(0,255,0),2)
        #print screenCon
        imageB = image.copy()
        cv2.drawContours(imageB, [screenCon], -1, (0,255,0),1)
            
        print ("Area: ")
        print area
        #cv2.imshow("contour",imageB)

        
        if area >= 30000.0: #Allow only contours of size 30000 or larger  
            print screenCon
            #Sort found contour coordinates 
            screenCon = sort_coord(screenCon)
            print ("Sorted")
            print screenCon         
            screenCon = np.array(screenCon,dtype="int")

            #Draw contour rectangle 
            rectangle = cv2.rectangle(imageB,(screenCon[0][0],screenCon[0][1]),(screenCon[2][0],(screenCon[2][1]-screenCon[1][1])*2+screenCon[2][1]),(0,255,0),1)
            cv2.imshow("rect",imageB)
            
            #Calculate centre of slide from x coordinates of coordinate 0 and 1 
            cropX = int((screenCon[1][0] - screenCon[0][0])/2) + screenCon[0][0] - 2
            #cropX = int((screenCon[1][0] - screenCon[0][0])/2)-((screenCon[1][0] - screenCon[0][0])/36)
            cropY = screenCon[0][1]+8 
            #YDist = 19
            YDist = int((screenCon[3][1]-screenCon[0][1]) / 8)*2+1

            #cv2.line(imageROI,(cropX,cropY),(cropX,cropY+50),(0,255,255),1)
            #cv2.imshow("line",imageROI)

            #Crop ROI of region around the spot - solid colour to measure variance between cameras 
            rROI = imageROI[cropY:cropY+50,cropX:cropX+5]
            gROI = imageROI[cropY+YDist+50:cropY+50*2+YDist,cropX:cropX+5]
            bROI = imageROI[cropY+YDist*2+50*2:cropY+50*3+YDist*2,cropX:cropX+5]

            cv2.imwrite("rROI.bmp",rROI)
            cv2.imwrite("gROI.bmp",gROI)
            cv2.imwrite("bROI.bmp",bROI)
            #Calculate mean and standard deviations of 50x5 rectangle 
            rMean,rStd = cv2.meanStdDev(rROI[:,:,2])
            gMean,gStd = cv2.meanStdDev(gROI[:,:,1])
            bMean,bStd = cv2.meanStdDev(bROI[:,:,0])

            #Print mean and standard deviations
            print("RGB Means: ")
            print(rMean,gMean,bMean)
            print("RGB Standard Deviations: ")
            print(rStd,gStd,bStd)
            
            #cv2.imshow("rROI",rROI)
            #cv2.imshow("gROI",gROI) 
            #cv2.imshow("bROI",bROI)

            #Crop ROI of region on left side of the slide 
            rROILeft = imageROI[cropY:cropY+50,screenCon[0][0]+6:screenCon[0][0]+6+6]
            gROILeft = imageROI[cropY+YDist+50:cropY+50*2+YDist,screenCon[0][0]+6:screenCon[0][0]+6+6]
            bROILeft = imageROI[cropY+YDist*2+50*2:cropY+50*3+YDist*2,screenCon[0][0]+6+2:screenCon[0][0]+6+6+2]
            cv2.imwrite("rROIL.bmp",rROILeft)
            cv2.imwrite("gROIL.bmp",gROILeft)
            cv2.imwrite("bROIL.bmp",bROILeft)
            #cv2.imshow("rLeft",rROILeft)
            #cv2.imshow("gLeft",gROILeft)
            #cv2.imshow("bLeft",bROILeft)
            rMeanL,rStdL = cv2.meanStdDev(rROILeft[:,:,2])
            gMeanL,gStdL = cv2.meanStdDev(gROILeft[:,:,1])
            bMeanL,bStdL = cv2.meanStdDev(bROILeft[:,:,0])
            print("\nLeft Edge Spot Means: ")
            print(rMeanL,gMeanL,bMeanL)
            print("Left Edge Spot Standard Deviations: ")
            print(rStdL,gStdL,bStdL)
            print("Standard Deviation Sum: ")
            print(rStdL+gStdL+bStdL)

            #Crop ROI of region on right side of the slide 
            rROIRight = imageROI[cropY:cropY+50,screenCon[2][0]-6-6:screenCon[2][0]-6]
            gROIRight = imageROI[cropY+YDist+50:cropY+50*2+YDist,screenCon[2][0]-6-6:screenCon[2][0]-6]
            bROIRight = imageROI[cropY+YDist*2+50*2:cropY+50*3+YDist*2,screenCon[2][0]-6-6-2:screenCon[2][0]-6-2]
            cv2.imwrite("rROIR.bmp",rROIRight)
            cv2.imwrite("gROIR.bmp",gROIRight)
            cv2.imwrite("bROIR.bmp",bROIRight)

            #cv2.imshow("rRight",rROIRight)
            #cv2.imshow("gRight",gROIRight)
            #cv2.imshow("bRight",bROIRight)
            rMeanR,rStdR = cv2.meanStdDev(rROIRight[:,:,2])
            gMeanR,gStdR = cv2.meanStdDev(gROIRight[:,:,1])
            bMeanR,bStdR = cv2.meanStdDev(bROIRight[:,:,0])
            print("\nRight Edge Spot Means: ")
            print(rMeanR,gMeanR,bMeanR)
            print("Right Edge Spot Standard Deviations: ")
            print(rStdR,gStdR,bStdR)
            print("Standard Deviation Sum: ")
            print(rStdR+gStdR+bStdR)

            #Crop ROI of region in centre of the slide 
            rROICentre = imageROI[cropY:cropY+50,cropX+8:cropX+14]
            gROICentre = imageROI[cropY+YDist+50:cropY+50*2+YDist,cropX+8:cropX+14]
            bROICentre = imageROI[cropY+YDist*2+50*2:cropY+50*3+YDist*2,cropX+8:cropX+14]
            cv2.imwrite("rROIC.bmp",rROICentre)
            cv2.imwrite("gROIC.bmp",gROICentre)
            cv2.imwrite("bROIC.bmp",bROICentre)

            rMeanC,rStdC = cv2.meanStdDev(rROICentre[:,:,2])
            gMeanC,gStdC = cv2.meanStdDev(gROICentre[:,:,1])
            bMeanC,bStdC = cv2.meanStdDev(bROICentre[:,:,0])
            print("\nCentre Edge Spot Means: ")
            print(rMeanC,gMeanC,bMeanC)
            print("Centre Edge Spot Standard Deviations: ")
            print(rStdC,gStdC,bStdC)
            print("Standard Deviation Sum: ")
            print(rStdC+gStdC+bStdC)
            sum
            #Plot subplot of all ROI 
            #image = mpimg.imread("rROI.bmp")
            pyplot.subplot(4,3,1),pyplot.axis("off"),pyplot.imshow(mpimg.imread("rROI.bmp")),pyplot.title("rROI")
            pyplot.subplot(4,3,2),pyplot.axis("off"),pyplot.imshow(mpimg.imread("gROI.bmp")),pyplot.title("gROI")
            pyplot.subplot(4,3,3),pyplot.axis("off"),pyplot.imshow(mpimg.imread("bROI.bmp")),pyplot.title("bROI")
            pyplot.subplot(4,3,4),pyplot.axis("off"),pyplot.imshow(mpimg.imread("rROIC.bmp")),pyplot.title("rROICentre")
            pyplot.subplot(4,3,5),pyplot.axis("off"),pyplot.imshow(mpimg.imread("gROIC.bmp")),pyplot.title("gROICentre")
            pyplot.subplot(4,3,6),pyplot.axis("off"),pyplot.imshow(mpimg.imread("bROIC.bmp")),pyplot.title("bROICentre")
            pyplot.subplot(4,3,7),pyplot.axis("off"),pyplot.imshow(mpimg.imread("rROIL.bmp")),pyplot.title("rROILeft")
            pyplot.subplot(4,3,8),pyplot.axis("off"),pyplot.imshow(mpimg.imread("gROIL.bmp")),pyplot.title("gROILeft")
            pyplot.subplot(4,3,9),pyplot.axis("off"),pyplot.imshow(mpimg.imread("bROIL.bmp")),pyplot.title("bROILeft")
            pyplot.subplot(4,3,10),pyplot.axis("off"),pyplot.imshow(mpimg.imread("rROIR.bmp")),pyplot.title("rROIRight")
            pyplot.subplot(4,3,11),pyplot.axis("off"),pyplot.imshow(mpimg.imread("gROIR.bmp")),pyplot.title("gROIRight")
            pyplot.subplot(4,3,12),pyplot.axis("off"),pyplot.imshow(mpimg.imread("bROIR.bmp")),pyplot.title("bROIRight")
            pyplot.show(block = False)


            #Draw all ROI onto image 
            cv2.rectangle(imageROI,(cropX+8,cropY),(cropX+14,cropY+50),(0,255,255),1)
            cv2.rectangle(imageROI,(cropX+8,cropY+YDist+50),(cropX+14,cropY+50*2+YDist),(0,255,255),1)
            cv2.rectangle(imageROI,(cropX+8,cropY+YDist*2+50*2),(cropX+14,cropY+50*3+YDist*2),(0,255,255),1)

            cv2.rectangle(imageROI,(cropX,cropY),(cropX+6,cropY+50),(0,255,0),1)
            cv2.rectangle(imageROI,(cropX,cropY+YDist+50),(cropX+6,cropY+50*2+YDist),(0,255,0),1)
            cv2.rectangle(imageROI,(cropX,cropY+YDist*2+50*2),(cropX+6,cropY+50*3+YDist*2),(0,255,0),1)

            cv2.rectangle(imageROI,(screenCon[0][0]+6,cropY),(screenCon[0][0]+6+6,cropY+50),(0,255,0),1)
            cv2.rectangle(imageROI,(screenCon[0][0]+6,cropY+YDist+50),(screenCon[0][0]+6+6,cropY+50*2+YDist),(0,255,0),1)
            cv2.rectangle(imageROI,(screenCon[0][0]+6+2,cropY+YDist*2+50*2),(screenCon[0][0]+6+6+2,cropY+50*3+YDist*2),(0,255,0),1)

            cv2.rectangle(imageROI,(screenCon[2][0]-6-6,cropY),(screenCon[2][0]-6,cropY+50),(0,255,0),1)
            cv2.rectangle(imageROI,(screenCon[2][0]-6-6,cropY+YDist+50),(screenCon[2][0]-6,cropY+50*2+YDist),(0,255,0),1)
            cv2.rectangle(imageROI,(screenCon[2][0]-6-6-2,cropY+YDist*2+50*2),(screenCon[2][0]-6-2,cropY+50*3+YDist*2),(0,255,0),1)

            cv2.imshow("ROI",imageROI)
            #Test Git
            #Plot mean / std data 

            #Save data into csv 

         
#Destroy / close all windows 
vc.release()
cv2.destroyWindow("preview")
cv2.destroyWindow("capture")