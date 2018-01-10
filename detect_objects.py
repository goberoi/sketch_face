import tensorflow as tf
import os
import cv2
import numpy as np

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Define what model weights to us, and where to find them.
BASE_PATH = 'object_detection'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_MODEL_WEIGHTS = os.path.join(BASE_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(BASE_PATH, 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Helper
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Create graph
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

# Definite input and output Tensors for detection_graph
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Start session
sess = tf.Session(graph=detection_graph)

def detect_objects(image):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_expanded = np.expand_dims(image, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    return (boxes, scores, classes, num)

print("got here")


if __name__ == "__main__":
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    process_frame = True

    while(True):
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if process_frame:
            print("detect objects")
            print(type(frame))
            (boxes, scores, classes, num) = detect_objects(frame)
            print(boxes)
            print(scores)
            print(classes)
            print(num)

        process_frame = (not process_frame)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("breaking")
            break

