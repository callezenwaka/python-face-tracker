# Currently not running on Mac OS X

import numpy as np
import argparse
import image_utils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

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

while True:
    (grabbed, frame) = camera.read()
    frame = image_utils.resize(frame, width=600)

    if not grabbed:
        break

    red = cv2.inRange(frame, redLower, redUpper)
    red = cv2.GaussianBlur(red, (3, 3), 0)

    (cnts, _) = cv2.findContours(red.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        rect = np.int32(cv2.cv.BoxPoints(cv2.minAreaRect(cnt)))
        cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Binary", red)

    time.sleep(0.025)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
