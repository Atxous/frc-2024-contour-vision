import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

lower_orange = np.array([14, 180, 210])
upper_orange = np.array([15, 182, 255])

# Load the image
img = cv2.imread("video_rings/0.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(img, lower_orange, upper_orange)
cv2.imshow("mask", mask)
cv2.waitKey(0)

# # get the pixel at [843, 273] and plot it
# pixel = img[273, 843]
# print(pixel)
# # show the pixel color (make sure it's 256 x 256)
# pixel_rgb = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_HSV2RGB)[0][0]

# # Create a 256x256 image with the pixel color
# color_img = np.full((256, 256, 3), pixel_rgb, dtype=np.uint8)

# # Display the color
# plt.imshow(color_img)
# plt.axis('off')  # to hide the axis
# plt.show()







# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (256, 256))

# r, g, b = cv2.split(img)

# pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()

# hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# h, s, v = cv2.split(hsv)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")

# axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Hue")
# axis.set_ylabel("Saturation")
# axis.set_zlabel("Value")
# plt.show()