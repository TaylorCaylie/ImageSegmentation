import cv2
import numpy as np
import matplotlib.pyplot as plt


# PART B
image = cv2.imread("sample2.jpg")
# convert to rgb
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# convert the image data to a 2dimensional array for the formatting of pixels and RGB
pixels = np.float32(image.reshape((-1, 3)))

# determine when to stop the clustering algo
# whenever 10 iterations have ran or an accuracy of 1.0 attached to 
# the epsilon has been reached
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#use kmeans_random_centers for random points
k = 5
ret, label, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert result back to uint8 and reshape to original image
centers = np.uint8(centers)
res = centers[label.flatten()]
res = res.reshape(image.shape)

plt.imshow(res)
plt.show()