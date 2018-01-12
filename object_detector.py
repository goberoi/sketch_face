import tensorflow as tf
import os
import cv2
import numpy as np
import time
from utils import FPS, WebcamVideoStream, convert_to_boxes_and_labels
from multiprocessing import Queue, Pool

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constants: mostly to define what model weights to use, and where to find them.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17' # Fastest, but poor accuracy.
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17' # Fast and reasonable accuracy.
#MODEL_NAME = 'faster_rcnn_resnet50_coco_2017_11_08' # Waaay too slow to even tell.
#MODEL_NAME = 'faster_rcnn_resnet50_lowproposals_coco_2017_11_08' # Still too slow.
PATH_TO_MODEL_WEIGHTS = os.path.join('models', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

class ObjectDetector:

    def __init__(self):
        self._detection_graph = None
        self._category_index = None
        self._sess = None



        # Create and load graph
        logger.info("Start loading model...")
        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            # Load model into memory
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL_WEIGHTS, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        logger.info("... done.")

        # Load label map
        logger.info("Start loading label map...")
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self._category_index = label_map_util.create_category_index(categories)
        logger.info("... done.")

        # Start self._session
        logger.info("Start TensorFlow session...")
        self._sess = tf.Session(graph=self._detection_graph)
        logger.info("... done.")


    # Call this for each image. Pass in an image as a numpy array
    def detect(self, image):
        # Define input and output Tensors for self._detection_graph
        image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_expanded = np.expand_dims(image, axis=0)
        # Actual detection
        (boxes, scores, classes, num) = self._sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        # Convert values returned from detector into a format that's easier to use for rendering
        (rect_points, class_names, class_colors) = convert_to_boxes_and_labels(
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self._category_index)
        return class_names, rect_points, class_colors

    
    # Render all detections, except thos in skip_classes, onto the given numpy array image
    def render(self, image, detections, skip_classes = []):
        height, width, channels = image.shape
        for class_name, rect, class_color in zip(*detections):
            class_name_without_percent = class_name[0].split(':')[0]
            if not class_name_without_percent in skip_classes:
                rect[0] = (int(rect[0][0] * width), int(rect[0][1] * height))
                rect[1] = (int(rect[1][0] * width), int(rect[1][1] * height))
                cv2.rectangle(image, rect[0], rect[1], class_color, 2)
                cv2.putText(image, class_name[0], rect[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, class_color, 2)
        return image


# Main function to demo/test when this is not used as a module.
if __name__ == "__main__":
    # Get a reference to webcam #0 (the default one)
    video_capture = WebcamVideoStream(src = 0, width = 480, height = 360).start()

    # Setup detector
    detector = ObjectDetector()
    
    # Track fps
    fps = FPS().start()

    while(True):
        # Grab a single frame of video
        frame = video_capture.read()

        # Convert image from BGR (used in opencv) to RGB (used by object_detection model)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

         # Detect!
        detections = detector.detect(frame)

        # Render
        frame = detector.render(frame, detections)

        # Convert the image back into BGR for opencv
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show in a window
        cv2.imshow("video", frame)

        # Wait 1 ms; cancel with a 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Track FPS
        fps.update()
    
    # Print time performance
    fps.stop()
    logger.info('Elapsed time (total): {:.2f}'.format(fps.elapsed()))
    logger.info('Approx. FPS: {:.2f}'.format(fps.fps()))

    # Cleanup
    video_capture.stop()
    cv2.destroyAllWindows()
