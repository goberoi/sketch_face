import face_recognition
import cv2
import pprint
import numpy as np
import random
import argparse
import time
#from queue import Queue
#from threading import Thread
import math

from quickdraw import QuickDraw
from utils import FPS, WebcamVideoStream, VideoOutputStream

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def face_landmarks_worker(input_q, output_q):
    logger.info('face worker: starting')

    while True:
        logger.debug('face worker: about to read from input_q')

        # Read input frame from queue
        frame = face_input_q.get()

        logger.debug('face worker: done reading input_q')

        # Detect facial landmarks
        logger.debug('face worker: about to detect face landmarks')
        t = time.time()
        face_landmarks = face_recognition.face_landmarks(frame)
        logger.debug('face worker: done detecting face landmarks in %s' % str(time.time() - t))

        face_output_q.put(face_landmarks)


class Sprite:
    def __init__(self, image, position=[0, 0], direction=[100, 100]):
        self._position = position
        self._direction = direction
        self._image = image
        self._out_of_bounds = False

    def update(self, elapsed):
        amplitude = 30
        frequency = 2
        self._position[0] += int(self._direction[0] * elapsed)
        self._position[1] += int(self._direction[1] * elapsed)
        # Check if it is out of bounds
        if self._position[0] > settings['width'] \
                or self._position[1] > settings['height'] \
                or self._position[0] < 0 \
                or self._position[1] < 0:
            self._out_of_bounds = True

    def render(self, canvas):
        QuickDraw.render(canvas, self._position[0], self._position[1], self._image, 0.5)
        pass

def _draw_face_landmark_points(face_landmarks, canvas):
    for face in face_landmarks: 
        for landmark, points in face.items():
            if landmark in ['top_lip']:
                continue
            np_points = np.array(points, dtype='int32')
            np_points *= settings['scale_frame']
            count = 1
            for point in np_points:
                count += 1
                cv2.circle(canvas, tuple(point), count, (0, 0, 255), 1)
    return

def compute_pose(face, canvas=None):
    image_points = np.array([
        face['nose_tip'][2],
        face['chin'][8],
        face['left_eye'][0],
        face['right_eye'][3],
        face['top_lip'][0],
        face['bottom_lip'][0]],
        dtype = 'double')
    image_points *= settings['scale_frame']

    for point in image_points:
        cv2.circle(canvas, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)

    # 3D model points.
    model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
            ])

    # Camera internals
    size = canvas.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
        )

    print("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # for p in image_points:
    #     cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(canvas, p1, p2, (255,0,0), 2)

    pose = [p2[0] - p1[0], p2[1] - p1[1]]
    print("pose: %s" % str(pose))

    return pose

if __name__ == '__main__':

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
    settings['height'] = 720
    settings['width'] = 1280
    settings['num_workers'] = 1
    settings['queue_size'] = 1

    # Setup face landmarks detection worker
#    face_input_q = Queue(settings['queue_size'])
#    face_output_q = Queue()
#    t = Thread(target=face_landmarks_worker, args=(face_input_q, face_output_q))
#    t.daemon = True
#    t.start()

    logger.info('workers loaded')

    # Get a reference to webcam #0 (the default one)
    video_capture = WebcamVideoStream(src = 0, 
                                      width = settings['width'], 
                                      height = settings['height']).start()

    # Setup some rendering related things
    canvas = None
    quickdraw = QuickDraw()
    sketch_images = {}
    line_color = (156,156,156)
    sprites = []

    # Track fps
    fps = FPS().start()

    while True:
        # Grab a single frame of video
        frame = video_capture.read()

        # Pick the background to draw on
        if settings['video']:
            canvas = frame.copy()
        else:
            canvas = np.zeros((settings['height'], settings['width'], 3), np.uint8)
            canvas[:, :, :] = (255, 255, 255)
        logger.debug('done setting up canvas')

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame of video to for faster face recognition processing
        frame = cv2.resize(frame, (0, 0), fx=(1/settings['scale_frame']), fy=(1/settings['scale_frame']))
        logger.debug('read, resized, and changed color scheme for one frame')

        # Pick random sketches every so often
        sketch_images['nose_bridge'] = quickdraw.get_random('nose')
        sketch_images['left_eye'] = quickdraw.get_random('eye')
        sketch_images['right_eye'] = sketch_images['left_eye']

        # Send task to async workers
#        face_input_q.put(frame)
#        logger.debug('put a frame on the input_q, now waiting on output_q')

        # Pull face landmark results and render them
#        face_landmarks = face_output_q.get()
#        logger.debug('got face landmarks from face_output_q')
        t = time.time()
        face_landmarks = face_recognition.face_landmarks(frame)
        logger.debug('done detecting face landmarks in %s' % str(time.time() - t))

        # Render face lines, and quickdraw sketch images
        logger.debug('worker: about to render face landmarks')
        t = time.time()
        for face in face_landmarks:
            # Draw landmarks
            for landmark, points in face.items():
                np_points = np.array(points, dtype='int32')
                np_points *= settings['scale_frame']

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
                        cv2.polylines(canvas, [np_points], close_polygon, line_color, 3)
                elif landmark in ['nose_tip']:
                    pass
                else:
                    cv2.polylines(canvas, [np_points], close_polygon, line_color, 3)
        logger.debug('worker: done rendering face landmarks %s' % str(time.time() - t))


        # Remove any sprites that have left the screen
        live_sprites = []
        for sprite in sprites:
            if not sprite._out_of_bounds:
                live_sprites.append(sprite)
        sprites = live_sprites

        # Create new sprites for each face if mouth is open
        for face in face_landmarks:
            # Compute if mouth is open if ratio of vertical open is nearly that of the horizontal mouth
            mouth_left = np.array(face['top_lip'][0])
            mouth_right = np.array(face['bottom_lip'][0])
            mouth_top = np.array(face['top_lip'][3])
            mouth_bottom = np.array(face['bottom_lip'][3])
            mouth_horizontal_distance = np.linalg.norm(mouth_right - mouth_left)
            mouth_vertical_distance = np.linalg.norm(mouth_bottom - mouth_top)
            if mouth_horizontal_distance and ((mouth_vertical_distance / mouth_horizontal_distance) > .8):
                # Compute head pose
                pose = compute_pose(face, canvas)
                mouth_center = np.array(face['top_lip'][3], dtype='int32') * settings['scale_frame']
                print("mouth center: %s" % str(mouth_center))
                sprite = Sprite(quickdraw.get_random('apple'),
                                position = mouth_center,
                                direction = pose)
                sprites.append(sprite)

        # Update sprites on a second canvas
        for sprite in sprites:
            sprite.update(fps.elapsed_since_last_update())
            sprite.render(canvas)

        # Display the resulting image
        cv2.imshow('Video', canvas)

        # Track FPS
        fps.update()

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Print time performance
    fps.stop()
    logger.info('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    logger.info('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    # Release handle to the webcam
    video_capture.stop()
    cv2.destroyAllWindows()
