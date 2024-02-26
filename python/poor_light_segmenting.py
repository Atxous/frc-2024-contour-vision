import cv2
import numpy as np
from color_constancy import equalization_and_cc, channel_enhancement_parallel
import matplotlib.pyplot as plt

# first we need to split the image into its 3 color components
ring = cv2.imread("video_rings/64.jpg") # replace with path to image
#size the image down to 1024 * 1024
ring = cv2.resize(ring, (512, 512))
ring_og = cv2.resize(ring.copy(), (1024, 1024))
b, g, r = cv2.split(ring)

# we now apply Ebner color constancy to the image

import time
start = time.time()
red = equalization_and_cc(r, iterations=1000)
green = equalization_and_cc(g, iterations=1000)
blue = equalization_and_cc(b, iterations=1000)
end = time.time()
print("Time taken: ", end - start)

start = time.time()
image = channel_enhancement_parallel(ring, iterations=1000)
end = time.time()
cv2.imshow("image", image)
print("Time taken: ", end - start)
# display the colors
cv2.imshow("red", red)
cv2.imshow("green", green)
cv2.imshow("blue", blue)
cv2.waitKey(0)

# we now merge the 3 color channels into a single image
ring = cv2.merge([blue, green, red])
# resize up to 1024 * 1024
ring = cv2.resize(ring, (1024, 1024))

# mask for orange
hsv = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)
# define range of orange color in HSV
lower_orange = np.array([5, 50, 50])
upper_orange = np.array([15, 255, 255])
# Threshold the HSV image to get only orange colors
mask = cv2.inRange(hsv, lower_orange, upper_orange)

hsv2 = cv2.cvtColor(ring_og, cv2.COLOR_BGR2HSV)
mask2 = cv2.inRange(hsv2, lower_orange, upper_orange)
# show the mask
cv2.imshow("mask", mask)
cv2.imshow("mask2", mask2)
# show
cv2.imshow("ring", ring)
cv2.waitKey(0)
#506, 211
# # get the pixel at [546, 427] and plot it
pixel = hsv[427, 546]
print(pixel)
# show the pixel color (make sure it's 256 x 256)
pixel_rgb = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_HSV2RGB)[0][0]

# Create a 256x256 image with the pixel color
color_img = np.full((256, 256, 3), pixel_rgb, dtype=np.uint8)

# Display the color
plt.imshow(color_img)
plt.axis('off')  # to hide the axis
plt.show()