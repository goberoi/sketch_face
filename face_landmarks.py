import face_recognition
import cv2
import pprint
import numpy as np
import quickdraw
import random

# Settings
process_nth_frame = 2
scale_frame = 3
blank_canvas = True
logging = True

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_landmarks_list = []
frame_count = 0
pp = pprint.PrettyPrinter(indent=4)
canvas = None
eye = random.choice(quickdraw.images['eye'])

# Helper
def log(msg):
    if logging:
        pp.pprint(msg)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Capture height and width of window
    height, width = frame.shape[:2]

    # Pick the background to draw on
    if blank_canvas:
        canvas = np.zeros((height,width,3), np.uint8)
        canvas[:, :, :] = (255, 255, 255)
    else:
        canvas = frame.copy()

    # Resize frame of video to for faster face recognition processing
    frame = cv2.resize(frame, (0, 0), fx=(1/scale_frame), fy=(1/scale_frame))

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if (frame_count % process_nth_frame) == 0:
        # Find all the faces and face encodings in the current frame of video
        face_landmarks = face_recognition.face_landmarks(rgb_frame)

    # Generate random features every so often
    if (frame_count % 10) == 0:
        eye = random.choice(quickdraw.images['eye'])

    # Increment counter to track nth frame to process
    frame_count = (frame_count + 1) % 10000000

    log(face_landmarks)

    # Display the results
    for face in face_landmarks:
        # Draw landmarks
        for landmark, points in face.items():
            np_points = np.array(points, dtype='int32')
            np_points *= scale_frame

            color = (156,156,156)
            close_polygon = False

            if landmark in ['left_eye', 'right_eye']:
                close_polygon = True
                centroid = np.mean(np_points, axis=0).astype('int')
#                cv2.circle(canvas, tuple(centroid), 5, color, 7)
                quickdraw.render(canvas, centroid[0], centroid[1], eye, 0.2)
            else:
                cv2.polylines(canvas, [np_points], close_polygon, color, 3)

    # Display the resulting image
    cv2.imshow('Video', canvas)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

