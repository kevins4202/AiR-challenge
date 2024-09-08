import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

def increaseContrast(image, alpha=1, beta=0): # alpha (contrast) 1.0-3.0, beta (brightness) 0-100
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)

def blur(img):
    # kernel = np.ones((5,5),np.float32)/25
    # img = cv.filter2D(img,-1,kernel)
    # img = cv.medianBlur(img,5)
    img = cv.GaussianBlur(img,(5,5),0)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.blur(img,(5,5))
    return img 

def canny(img):
    edges = cv.Canny(img, 100, 200)
    return edges

def contours(img):
    # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, thresh = cv.threshold(img, 255, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow('thresh', thresh)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return img, contours, thresh


cap = cv.VideoCapture('p3.mp4')
 
while cap.isOpened():
    kernel = np.ones((5,5),np.uint8)
    ret, img = cap.read()
    frame= img.copy()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = increaseContrast(img, 3 , 15)

    img = canny(img)
    plt.imshow(img, cmap='gray')
    # plt.show()
    # for i in range(10):
    #     img = blur(img)
    img = cv.dilate(img,kernel,iterations = 4)
    
    img, cnts, threshold = contours(img)

    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv.drawContours(img, cnts, -1, (0,255,0), 3)
    plt.imshow(img)
    # plt.show()

    centers = set()
    cnts = list(cnts)
    # print(cnts)
    for i in range(len(cnts)):
        c = cnts[i]
        # compute the center of the contour
        if cv.contourArea(c) < 5000 or cv.contourArea(c) > 100000:
            #remove
            continue
        M = cv.moments(c)
        
        if M['m00'] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            if (cX, cY) in centers:
                continue
            cv.circle(frame, (cX, cY), 4, (255, 255, 255), -1)
            cv.putText(frame, f'({cX}, {cY})', (cX - 20, cY - 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # show the image
            centers.add((cX, cY))
    cv.drawContours(frame, cnts, -1, (255,255,255), 2) 
    cv.imshow('frame', frame)
    # plt.imshow(contour_img, cmap='gray')
    # plt.show()

    if cv.waitKey(1) == ord('q'):
        break
    # while cv.waitKey(1) != ord('x'):
    #     pass
cap.release()
cv.destroyAllWindows()