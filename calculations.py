import numpy as np
import cv2

def get_coords_of_low_and_high_mask(mask):
    #find the coordinates of the lowest and highest pixel in shape CHW where the mask is 1
    #returns a tuple of tuples
    indices = np.where(mask == 255)
    lowest_x = indices[1].min()
    lowest_y = indices[0].min()
    highest_x = indices[1].max()
    highest_y = indices[0].max()
    return ((lowest_x, lowest_y), (highest_x, highest_y))

def are_contours_close(contour1, contour2, threshold):
    # returns true if the contours are close enough to be connected
    # compares the distance between the highest and lowest points of the contours
    x1,y1,w1,h1 = cv2.boundingRect(contour1)
    x2,y2,w2,h2 = cv2.boundingRect(contour2)
    # take the distance formula between high high, low low, high low, and low high
    # if any of these distances are less than the threshold, return true
    distances = [np.linalg.norm(np.array([x1+w1, y1+h1]) - np.array([x2+w2, y2+h2])),
                 np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])),
                 np.linalg.norm(np.array([x1+w1, y1]) - np.array([x2+w2, y2+h2])),
                 np.linalg.norm(np.array([x1, y1+h1]) - np.array([x2, y2+h2]))]
    for distance in distances:
        if distance < threshold:
            return True
    return False
    
def connect_biggest_contour_with_close_contours(contours, threshold = 10):
    #returns a list of contours where the biggest contour is connected to other close contours
    #threshold is the maximum distance between the contours
    biggest_contour = max(contours, key = cv2.contourArea)
    connected_contours = [biggest_contour]
    for contour in contours:
        if contour is biggest_contour:
            continue
        elif are_contours_close(biggest_contour, contour, threshold):
            connected_contours.append(contour)

    return connected_contours