import face_recognition
import cv2
import pprint
import numpy as np
import random
import argparse
import time
from queue import Queue
from threading import Thread
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
    def __init__(self, image, position=[0, 0], direction=(100, 100)):
        self._position = position
        self._direction = direction
        self._image = image

    def update(self, elapsed):
        amplitude = 30
        frequency = 2
        self._position[0] += int(self._direction[0] * elapsed)
        self._position[1] += int(self._direction[1] * elapsed)
        if self._position[0] > settings['width'] \
                or self._position[1] > settings['height'] \
                or self._position[0] < 0 \
                or self._position[1] < 0:
            self._position = [0, 0]

    def render(self, canvas):
        QuickDraw.render(canvas, self._position[0], self._position[1], self._image, 0.5)
        pass

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
    sprites.append(Sprite(quickdraw.get_random('apple')))
#    video_output = VideoOutputStream().start()

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

        # Update sprites on a second canvas
        for sprite in sprites:
            sprite.update(fps.elapsed_since_last_update())
            sprite.render(canvas)

        # Display the resulting image
        cv2.imshow('Video', canvas)
#        video_output.write(frame)

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