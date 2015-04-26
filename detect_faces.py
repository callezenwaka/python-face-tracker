# Code based almost entirely on books
# published by Dr. Adrian Rosebrock
# in his books at https://www.pyimagesearch.com/

from face_detector import FaceDetector
import argparse
import cv2

# Parse arguments pertaining to a serialized classifier
# that's been trained to recognize faces within images
# as well as an input image containing faces you want
# to detect
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,
                help = "path to where the face cascade resides")
ap.add_argument("-i", "--image", required = True,
                help = "path to where the image file resides")
args = vars(ap.parse_args())

# Use OpenCV to parse the input image file and detect
# faces within it using the specified classifier
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fd = FaceDetector(args["face"])
faceRects = fd.detect(gray,
                      scaleFactor = 1.1,
                      minNeighbors = 8,
                      minSize = (30, 30))
print "I found %d face(s)" % (len(faceRects))

# Alter the input image so that it can be displayed.
for (x, y, w, h) in faceRects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the altered image in a window until the user
# presses a key to end the program.
cv2.imshow("Faces", image)
cv2.waitKey(0)
