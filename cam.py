from face_detector import FaceDetector
import image_utils
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-v" "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

# Create a FaceDetector using a serialized classifier
# that's trained specifically for frontal face detection
fd = FaceDetector(args["face"])

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

# Loop over all the frames in the video and
# detect the faces in each individual frame
# until we either run out of frames (as would
# happen with a video file), or the user opts
# to exit the program manually
while True:
    (grabbed, frame) = camera.read()

    # See if we ran out of frames in a
    # video file that was specified via
    # a command line argument
    if args.get("video") and not grabbed:
        break

    # Resize the frame to have a width of only
    # 600px solely for performance reasons
    # associated with real-time face detection
    # using a nice little helper function
    frame = image_utils.resize(frame, width=600)

    # Convert the frame to grayscale prior to analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Attempt to detect faces in the grayscale
    # version of the current video frame
    faceRects = fd.detect(gray, scaleFactor=1.2,
                          minNeighbors=5, minSize=(30, 30))

    # Before modifying the current frame, we'll clone
    # it just in case we decide we want to do some
    # additional processing on the original
    frameClone = frame.copy()

    # We'll draw a green bounding box around each of
    # the faces that we detected within our video
    # frame by modifying the cloned copy
    for (fX, fY, fW, fH) in faceRects:
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 255, 0), 2)

    # We'll display the output of the cloned video
    # frame immediately after we've detected faces
    # within it and drawn the bounding boxes
    cv2.imshow("Face", frameClone)

    # We'll allow the user to quit the program either
    # by hitting the 'q' key rather than just Ctrl+C,
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the reference to the camera
# before exiting the program
camera.release()

# Close any currently open windows that
# were created by OpenCV
cv2.destroyAllWindows()
