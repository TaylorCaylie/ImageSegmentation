import cv2
import matplotlib.pyplot as plt
import numpy as np

# Part C 

# read in image
image = cv2.imread("sample2.jpg")

# convert to rgb
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# convert to greyscale
greyimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# convert to a binary threshold
_, binaryimage = cv2.threshold(greyimage, 220, 255, cv2.THRESH_BINARY_INV)

# get edge
edge = cv2.Canny(binaryimage, 60, 180)

# find contours from the image generated above
contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# execute the contours
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

# apply the countours to a blank canvas
blank = np.zeros(image.shape, dtype='uint8')
contoursimage = cv2.drawContours(blank, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

# show both the blank canvas countours and countours over image
plt.imshow(contoursimage)
plt.show()

plt.imshow(image)
plt.show()