import numpy as np
import skimage.color
import skimage.io
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# read in image
image = cv2.imread("sample2.jpg")

# convert to rgb
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# convert to greyscale
greyimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# blur the image
blurredimage = cv2.GaussianBlur(greyimage,(5,5),cv2.BORDER_DEFAULT)

# create histogram based on read in image
histogram, bin_edges = np.histogram(greyimage, bins=256, range=(0, 1))

fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)
plt.title("Grey Level Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
plt.show()

# PART A i

# create binary mask to extract objects
binary = cv2.adaptiveThreshold(greyimage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

fig, ax = plt.subplots()
plt.imshow(binary, cmap='gray')
plt.show()


# PART A ii
thresholds = threshold_multiotsu(image=greyimage, classes=4)
regions = np.digitize(greyimage, bins=thresholds)


fig, ax = plt.subplots()
plt.imshow(regions)
plt.show()


