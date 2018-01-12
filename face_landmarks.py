import face_recognition
import cv2
import pprint
import numpy as np
import quickdraw
import random
import argparse
from utils import FPS, WebcamVideoStream
import detect_objects

# Settings via command line args or defaults
parser = argparse.ArgumentParser()
parser.add_argument("--video", 
                    help="show the camera's video feed in the background",
                    action="store_true")
parser.add_argument("--sketch", 
                    help="show facial features as hand drawn images from the quick-draw dataset",
                    action="store_true")
settings = vars(parser.parse_args())

settings['process_nth_frame'] = 1
settings['scale_frame'] = 4

# Initialize some variables
face_landmarks_list = []
frame_count = 0
pp = pprint.PrettyPrinter(indent=4)
canvas = None
sketch_images = None
width = 1280
height = 720

# Get a reference to webcam #0 (the default one)
video_capture = WebcamVideoStream(src = 0, width = width, height = height).start()

# Load object detection
detect_objects.init()

# Track fps
fps = FPS().start()

while True:
    # Grab a single frame of video
    frame = video_capture.read()

    # Pick the background to draw on
    if settings['video']:
        canvas = frame.copy()
    else:
        canvas = np.zeros((height,width,3), np.uint8)
        canvas[:, :, :] = (255, 255, 255)

    # Detect
    detections = detect_objects.detect(frame)

    # Render boxes
    canvas = detect_objects.render(canvas, detections, skip_classes = ['person'])

    # Resize frame of video to for faster face recognition processing
    frame = cv2.resize(frame, (0, 0), fx=(1/settings['scale_frame']), fy=(1/settings['scale_frame']))

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    if (frame_count % settings['process_nth_frame']) == 0:
        # Find all the faces and face encodings in the current frame of video
        face_landmarks = face_recognition.face_landmarks(rgb_frame)

    # Pick random sketches every so often
    if ((frame_count % 10) == 0) or (not sketch_images):
        sketch_images = { name : random.choice(quickdraw.images[name]) for name in ['eye', 'mouth', 'nose']}
        sketch_images['nose_bridge'] = sketch_images['nose']
        sketch_images['left_eye'] = sketch_images['eye']
        sketch_images['right_eye'] = sketch_images['eye']

    # Increment counter to track nth frame to process
    frame_count = (frame_count + 1) % 10000000

    # Display the results
    for face in face_landmarks:
        # Draw landmarks
        for landmark, points in face.items():
            np_points = np.array(points, dtype='int32')
            np_points *= settings['scale_frame']

            color = (156,156,156)
            close_polygon = False

            if landmark in ['left_eye', 'right_eye', 'nose_bridge']:
                close_polygon = True
                centroid = np.mean(np_points, axis=0).astype('int')
                sketch_image_scale = 0.2
                if landmark in ['nose_bridge']:
                    sketch_image_scale = 0.5
#                cv2.circle(canvas, tuple(centroid), 5, color, 7)
                if settings['sketch']:
                    quickdraw.render(canvas, centroid[0], centroid[1], sketch_images[landmark], 0.2)
                else:
                    cv2.polylines(canvas, [np_points], close_polygon, color, 3)
            elif landmark in ['nose_tip']:
                pass
            else:
                cv2.polylines(canvas, [np_points], close_polygon, color, 3)

    # Display the resulting image
    cv2.imshow('Video', canvas)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Track FPS
    fps.update()

# Print time performance
fps.stop()
print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

# Release handle to the webcam
video_capture.stop()
cv2.destroyAllWindows()

