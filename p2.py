import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def blur(img):
    # kernel = np.ones((5,5),np.float32)/25
    # img = cv.filter2D(img,-1,kernel)
    # img = cv.medianBlur(img,5)
    img = cv.GaussianBlur(img,(7,7),0)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.blur(img,(5,5))
    return img 

def canny(img):
    edges = cv.Canny(img, 100, 200)
    return edges

def contours(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 255, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0,255,0), 3)

    return img, contours, thresh


cap = cv.VideoCapture('p2.mp4')
 
while cap.isOpened():
    ret, img = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    for i in range(1):
        img = blur(img)

    # plt.imshow(img)

    image_copy = img.copy()
    lower_green = np.array([0,0,0])  
    upper_green = np.array([125,160,150])

    mask = cv.inRange(image_copy, lower_green, upper_green)

    masked_image = np.copy(image_copy)
    masked_image[mask != 0] = [255,255,255]

    canny_img = canny(masked_image)

    # img2 =  cv.cvtColor(canny_img,cv.COLOR_GRAY2RGB)

    # contour_img, cnts, threshold = contours(img2)

    centers = set()

    # for c in cnts:
    #     # compute the center of the contour
    #     M = cv.moments(c)
        
    #     if M['m00'] != 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #         # draw the contour and center of the shape on the image
    #         if (cX, cY) in centers:
    #             continue
    #         cv.circle(img2, (cX, cY), 4, (255, 255, 255), -1)
    #         cv.putText(img2, f'({cX}, {cY})', (cX - 20, cY - 20),
    #                 cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    #         # show the image
    #         centers.add((cX, cY))
 
    cv.imshow('frame', canny_img)

    if cv.waitKey(1) == ord('q'):
        break
    # while cv.waitKey(1) != ord('x'):
    #     pass
cap.release()
cv.destroyAllWindows()