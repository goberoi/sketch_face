import tensorflow as tf
import os
import cv2
import numpy as np
import time
from utils import FPS, WebcamVideoStream

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Constants: mostly to define what model weights to use, and where to find them.
BASE_PATH = 'object_detection'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_MODEL_WEIGHTS = os.path.join(BASE_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(BASE_PATH, 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Gobals: so we can initialze and load weights just once
detection_graph = None
category_index = None
sess = None

# Initializer. Please call me first to laod the graph, category names, and start a session.
def init():
    global detection_graph
    global category_index
    global sess

    # Create and load graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        # Load model into memory
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_MODEL_WEIGHTS, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Load label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Start session
    sess = tf.Session(graph=detection_graph)

# Call this for each image. Pass in an image as a numpy array.
def detect_and_visualize(image):
    start_time = time.clock()
    global detection_graph
    global category_index
    global sess
    # Create a small frame for faster object detection
    small_image = image.copy()
    small_image = cv2.resize(small_image, (0, 0), fx=(1/1), fy=(1/1))
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    small_image_expanded = np.expand_dims(small_image, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: small_image_expanded})
    # Visualize by drawing on the numpy array image
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    # Log time elapsed
    print("detect time: " + str(time.clock() - start_time))
    # We modify the image in place, but return the image anyways
    return image

# Main function to demo/test when this is not used as a module.
if __name__ == "__main__":
    # Get a reference to webcam #0 (the default one)
    video_capture = WebcamVideoStream(src = 0, width = 480, height = 360).start()

    # Setup detector
    init()
    
    # Used to process every other frame
    process_frame = True

    # Track fps
    fps = FPS().start()

    while(True):
        # Grab a single frame of video
        frame = video_capture.read()

        # Detect!
        if process_frame:
            detect_and_visualize(frame)
        process_frame = (not process_frame)

        # Show in a window
        cv2.imshow("video", frame)

        # Wait 1 ms; cancel with a 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Track FPS
        fps.update()


