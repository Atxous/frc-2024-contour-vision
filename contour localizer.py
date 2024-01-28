import cv2
from calculations import connect_biggest_contour_with_close_contours

LOWER_RANGE = (1, 160, 170) # lower range of orange in HSV
UPPER_RANGE = (20, 255, 255) # upper range of orange in HSV
SCALING_FACTOR = 0.25 # factor to scale down the image by (for more efficient processing)
RESCALING_FACTOR = int(1/SCALING_FACTOR) # factor to scale up the image by
LINE_COLOR = (0,255,0) # color of the line that goes through the center of the ring
PRINT_CENTER = True # whether or not to print the center of the ring to the console
CONNECT_CONTOURS = False # whether or not to connect contours that are close to each other (in case there is obstruction of the ring)
THRESHOLD = 10 # threshold for how close contours have to be to be connected (in pixels)

if CONNECT_CONTOURS:
    assert THRESHOLD > 0, "Threshold must be greater than 0 if CONNECT_CONTOURS is True"

if __name__ == "__main__":
    video = cv2.VideoCapture(0) # 0 is the default camera
    cv2.namedWindow("tracker", cv2.WINDOW_NORMAL)
    
    while video.isOpened():
        #get frame from camera
        ret, frame = video.read()

        frame = cv2.resize(frame, (0,0), fx=SCALING_FACTOR, fy=SCALING_FACTOR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        light_orange = LOWER_RANGE
        dark_orange = UPPER_RANGE
        mask = cv2.inRange(hsv, light_orange, dark_orange)

        # find all contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            if CONNECT_CONTOURS:
                # connect the biggest contour with other close contours
                contours = connect_biggest_contour_with_close_contours(contours, THRESHOLD)
                contours = [cv2.boundingRect(c) for c in contours]
                
                leftmost_contour = min(contours, key=lambda c: c[0])
                highest_contour = min(contours, key=lambda c: c[1])
                rightmost_contour = max(contours, key=lambda c: c[0] + c[2])
                lowest_contour = max(contours, key=lambda c: c[1] + c[3])
                
                center = ((leftmost_contour[0] + rightmost_contour[0] + rightmost_contour[2]) // 2 * RESCALING_FACTOR, (highest_contour[1] + lowest_contour[1] + lowest_contour[3]) // 2 * RESCALING_FACTOR)
            else:
                c = max(contours, key = cv2.contourArea)
                x,y,w,h = cv2.boundingRect(c)
                
                center = (((x*2 + w) // 2) * RESCALING_FACTOR, ((y*2 + h) // 2) * RESCALING_FACTOR)
                
            frame = cv2.resize(frame, (0,0), fx=RESCALING_FACTOR,fy=RESCALING_FACTOR)
            
            #get the center and draw an axis through it
            cv2.line(frame, (center[0] - 10, center[1]), (center[0] + 10, center[1]), LINE_COLOR, 2)
            cv2.line(frame, (center[0], center[1] - 10), (center[0], center[1] + 10), LINE_COLOR, 2)
            if PRINT_CENTER:
                print("Center: ", center)
        else:
            frame = cv2.resize(frame, (0,0), fx=RESCALING_FACTOR, fy=RESCALING_FACTOR)
            
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video.release()
    cv2.destroyAllWindows()

