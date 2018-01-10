import tensorflow as tf
import os

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
# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Start session
sess = tf.Session(graph=detection_graph)

print("got here")
