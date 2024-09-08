# %matplotlib ipympl

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('p1.png')
frame = img.copy()
assert img is not None, "file could not be read, check with os.path.exists()"
plt.figure(figsize=(12, 6))
plt.imshow(img)

def blur(img):
    # kernel = np.ones((5,5),np.float32)/25
    # img = cv.filter2D(img,-1,kernel)
    # img = cv.medianBlur(img,5)
    img = cv.GaussianBlur(img,(7,7),0) #use gaussian blur
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.blur(img,(5,5))
    return img

for i in range(3):
    img = blur(img) #blur three times

def increaseContrast(image, alpha=1, beta=0): # alpha (contrast) 1.0-3.0, beta (brightness) 0-100
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta) 

image_copy = img.copy()#mask green background
lower_green = np.array([0,0,0])  
upper_green = np.array([125,175,150])

mask = cv.inRange(image_copy, lower_green, upper_green)

masked_image = np.copy(image_copy)
masked_image[mask != 0] = [255,255,255]

def canny(img):
    edges = cv.Canny(img, 100, 200)
    return edges

def contours(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 255, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img, contours, 2, (0,255,0), 3)
    cv.drawContours(img, contours, -1, (0,255,0), 3)

    return img, contours, thresh

canny_img = canny(masked_image) #canny edge detection

img2 =  cv.cvtColor(canny_img,cv.COLOR_GRAY2RGB)

contour_img, cnts, threshold = contours(img2) #draw contours
frame = cv.drawContours(frame, cnts, -1, (0,0,0), 3)
centers = set()

for c in cnts:
    # compute the center of the contour
    M = cv.moments(c)
     
    if M['m00'] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        if (cX, cY) in centers:
            continue
        cv.circle(frame, (cX, cY), 4, (0, 0, 0), -1)
        cv.putText(frame, f'({cX}, {cY})', (cX - 20, cY - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        # show the image
        centers.add((cX, cY))
        print(f"x: {cX} y: {cY}")
    else:
        print("fail")
     
plt.imshow(img2, cmap='gray')
plt.show()
#save image
cv.imwrite('p1_output.png', frame)