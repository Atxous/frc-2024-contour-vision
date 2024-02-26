import cv2
import numpy as np
import concurrent.futures
from multiprocessing import Pool

# Apply histogram equalization and color constancy to an image
def equalization_and_cc(img, percentage=0.0005, iterations=100):
    # Normalize to [0, 1]
    c_i = cv2.equalizeHist(img).astype(np.float32) / 255

    # Initialize a_i to be the same as c_i
    # a_i is the average map
    # c_i is the base image
    a_i = c_i.copy()

    # Define kernel for averaging
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.float32) / 4

    # Perform convolution for averaging neighbors
    for _ in range(iterations):
        avg = cv2.filter2D(a_i, -1, kernel)

        # Update a_i using vectorized operations
        a_i = np.add(np.multiply(c_i, percentage), np.multiply(avg, (1 - percentage)))

    # Clip values
    c_i /= (2 * a_i + 1e-6)

    # Normalize between [0, 255]
    c_i = np.clip(c_i * 255, 0, 255).astype(np.uint8)

    return c_i

def channel_enhancement_parallel(image, percentage = 0.0005, iterations = 100):
    # split the image into its 3 color components and apply histogram equalization to each color channel
    channels = cv2.split(image)
    processed_channels = [None] * 3
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(equalization_and_cc, channel, percentage, iterations): i for i, channel in enumerate(channels)}
        for future in concurrent.futures.as_completed(futures):
            channel_index = futures[future]
            processed_channel = future.result()
            processed_channels[channel_index] = processed_channel
            
    return cv2.merge(processed_channels)



