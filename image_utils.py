import numpy as np
import cv2

def translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Return the translated image
    return shifted

def rotate(image, angle, center = None, scale = 1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # Set the center of the image if necessary
    if center is None:
        center = (w / 2, h / 2)

    # Rotate the image
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Returned the rotated image
    return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # Initialize the dimensions of the image to be resized
    # and grab the size of the image
    dim = None
    (h, w) = image.shape[:2]

    # If both the height and width are None,
    # then return the original image
    if height is None and width is None:
        return image

    # If the width is None, then calculate
    # the ratio of the height and construct
    # the dimensions
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)

    # Otherwise, the height is None, so we'll
    # similarly calculate the ratio of the
    # height and construct the dimensions
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    # Resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # Return the resized image
    return resized
