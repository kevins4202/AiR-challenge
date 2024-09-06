# %matplotlib ipympl

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('p1.png')
assert img is not None, "file could not be read, check with os.path.exists()"
plt.figure(figsize=(12, 6))
plt.imshow(img)

def blur(img):
    # kernel = np.ones((5,5),np.float32)/25
    # img = cv.filter2D(img,-1,kernel)
    # img = cv.medianBlur(img,5)
    img = cv.GaussianBlur(img,(7,7),0)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.blur(img,(5,5))
    return img

for i in range(3):
    img = blur(img)

def increaseContrast(image, alpha=1, beta=0): # alpha (contrast) 1.0-3.0, beta (brightness) 0-100
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)

# img = increaseContrast(img, 1, 30)

image_copy = img.copy()
lower_green = np.array([0,0,0])  
upper_green = np.array([125,175,150])

mask = cv.inRange(image_copy, lower_green, upper_green)

# plt.imshow(mask, cmap='gray')

masked_image = np.copy(image_copy)
masked_image[mask != 0] = [255,255,255]
# plt.imshow(masked_image)

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

# contour_img, cnts, threshold = contours(masked_image)
# blank = np.zeros(threshold.shape[:2], 
#                  dtype='uint8')
 
# cv.drawContours(blank, cnts, -1, 
#                 (255, 0, 0), 1)


# plt.imshow(blank)
canny_img = canny(masked_image)
# plt.title("Canny Image")
# plt.imshow(canny_img, cmap='gray')
# cv.imwrite('canny.png', canny_img)

# img2 = cv.imread('canny.png')
img2 =  cv.cvtColor(canny_img,cv.COLOR_GRAY2RGB)

contour_img, cnts, threshold = contours(img2)
# plt.title("Contour Image")
# plt.imshow(img2, cmap='gray')

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
        cv.circle(img2, (cX, cY), 4, (255, 255, 255), -1)
        cv.putText(img2, f'({cX}, {cY})', (cX - 20, cY - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # show the image
        centers.add((cX, cY))
        print(f"x: {cX} y: {cY}")
    else:
        print("fail")
     
plt.imshow(img2, cmap='gray')
plt.show()