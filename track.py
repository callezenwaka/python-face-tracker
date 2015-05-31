import numpy as np
import argparse
import image_utils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

# Define the upper and lower limits of our RGB
# colors for a Red Swingline stapler keeping in 
# mind that OpenCV uses RGB in reverse order
redLower = np.array([0, 0, 100], dtype="uint8")
redUpper = np.array([50, 50, 255], dtype="uint8")

# If the user didn't specify a video file,
# then we'll assume that we should try
# to capture video directly from the
# user's computer webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# If the user did specify a video file,
# then we'll use that as our video source
else:
    camera = cv2.VideoCapture(args["video"])

# Loop over the video frames one at a time
while True:
    # Grab the frame from the camera as a 
    # nice tuple that not only gives us the
    # frame, but also provides a boolean 
    # value indicating whether or not the
    # frame was successfully grabbed
    (grabbed, frame) = camera.read()

    if not grabbed:
        break

    # Resize the frame to a smaller 600px width
    frame = image_utils.resize(frame, width=600)

    # Attempt to find shades of the defined Red
    # color with each frame that's grabbed, and
    # get a result that's a thresholded image
    # with the pixels which are in the range
    # set to white and the rest of the pixels
    # set to black
    red = cv2.inRange(frame, redLower, redUpper)

    # Blur the thresholded image in order to make
    # the search for contours within it more accurate
    red = cv2.GaussianBlur(red, (3, 3), 0)

    # Find the contours within the thresholded image
    # using a copy of the image since the findContours
    # function effectively destroys the NumPy array
    # representing the image that's passed into it
    (cnts, _) = cv2.findContours(red.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    # If at least one contour was found in the image,
    # then we'll sort the list of contours in reverse
    # order for the purpose of retrieving the largest
    # contour that was found in the thresholded image
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        # Compute the minimum bounding box surrounding the largest
        # contour in the thresholded iamge, and then reshape that 
        # bounding box to be a list of points 
        rect = np.int32(cv2.cv.BoxPoints(cv2.minAreaRect(cnt)))

        # Draw a green bounding box around the largest contour
        # in the original frame that was grabbed from the video 
        # feed, and not the thresholded version of the frame
        cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

    # Display the the modified copy of the original video
    # frame with a green bounding box surrounding the stapler
    cv2.imshow("Tracking", frame)

    # Display the thresholded version of the video frame
    # in order to demonstrate how we were able to detect
    # the contours in the original frame 
    cv2.imshow("Binary", red)

    # Optionally slow down the processing since many newer
    # systems are capable of parsing well over 32 frames/second
    time.sleep(0.025)

    # Allow users to quit uxing the "q" key as well as the typical
    # Ctrl-C approach 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Destroy the reference to the camera and close
# any windows that OpenCV may have opened.
camera.release()
cv2.destroyAllWindows()
