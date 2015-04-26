# Code based almost entirely on books
# published by Dr. Adrian Rosebrock
# in his books at https://www.pyimagesearch.com/

import cv2

class FaceDetector:
    # A constructor that accepts the location of a
    # serialized classifier that has been trained
    # to recognize faces within an image
    def __init__(self, faceCascadePath):
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

    # A method that leverages the OpenCV library in conjunction
    # with OpenCV classifier instance created from the serialized
    # classifer file provided by the user to detect faces via a
    # set of parameters that can be tweaked to meet your needs.
    # For instance, setting minNeighbors to too a high a value
    # will cause the algorithm to be far pickier about detecting
    # faces and will result in not detecting some faces, whereas
    # setting too low will result in detecting false positives
    # within the image that's being analyzed.
    def detect(self,
               image,
               scaleFactor = 1.1,
               minNeighbors = 5,
               minSize = (30, 30)):

        # Creates a set of rectangles expressed as tuples to
        # indicate the location of all of the faces detected
        # within the input image.
        rects = self.faceCascade.detectMultiScale(image,
                                                  scaleFactor = scaleFactor,
                                                  minNeighbors = minNeighbors,
                                                  minSize = minSize,
                                                  flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

        return rects