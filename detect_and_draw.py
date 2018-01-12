import face_recognition
import cv2
import pprint
import numpy as np
import quickdraw
import random
import argparse
import time
from queue import Queue
from threading import Thread

from utils import FPS, WebcamVideoStream
from object_detector import ObjectDetector

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def worker(input_q, output_q):
    logger.info('worker: starting')

    # Initialize some variables
    face_landmarks_list = []
    frame_count = 0
    canvas = None
    sketch_images = None
    detector = ObjectDetector()

    logger.debug('worker: done initializing variables')

    while True:
        logger.debug('worker: about to read from input_q')

        # Read input frame from queue
        frame = input_q.get()

        logger.debug('worker: done reading input_q')

        # Pick the background to draw on
        if settings['video']:
            canvas = frame.copy()
        else:
            canvas = np.zeros((settings['height'], settings['width'], 3), np.uint8)
            canvas[:, :, :] = (255, 255, 255)

        logger.debug('worker: done setting up canvas')

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame of video to for faster face recognition processing
        frame = cv2.resize(frame, (0, 0), fx=(1/settings['scale_frame']), fy=(1/settings['scale_frame']))

        logger.debug('worker: about to detect objects')
        t = time.time()

        # Detect objects
        detections = detector.detect(frame)

        logger.debug('worker: done detecting objects in %s' % str(time.time() - t))

        # Detect facial landmarks
        face_landmarks = face_recognition.face_landmarks(frame)
            
        # Render boxes
        canvas = detector.render(canvas, detections, skip_classes = ['person'])

        # Pick random sketches every so often
        if (not sketch_images) or (random.randint(1,100) < 10):
            sketch_images = { name : random.choice(quickdraw.images[name]) for name in ['eye', 'mouth', 'nose']}
            sketch_images['nose_bridge'] = sketch_images['nose']
            sketch_images['left_eye'] = sketch_images['eye']
            sketch_images['right_eye'] = sketch_images['eye']

        # Render face lines, and quickdraw sketch images
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

        output_q.put(canvas)


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

    # Setup multithreading stuff
    input_q = Queue(settings['queue_size'])
    output_q = Queue()
    for i in range(settings['num_workers']):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    logger.info('workers loaded')

    # Get a reference to webcam #0 (the default one)
    video_capture = WebcamVideoStream(src = 0, 
                                      width = settings['width'], 
                                      height = settings['height']).start()

    logger.info('video capture start')

    # Track fps
    fps = FPS().start()

    while True:
        # Grab a single frame of video
        frame = video_capture.read()

        logger.debug('read one frame')

        # Send task to async workers
        input_q.put(frame)

        logger.debug('put a frame on the input_q, now waiting on output_q')

        # Pull any available results from async workers
        if output_q.empty():
            canvas = frame
        else:
            canvas = output_q.get()
            logger.debug('got frame from output_q')

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
