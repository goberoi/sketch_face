

## Notes

### Download pretrained models

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


### Coco dataset cateagories that are relevant:

apple
banana
sandwich
bottle
cup
fork
spoon
knife
wine glass
cell phone
book
clock
keyboard
laptop
mouse
scissors
backpack
handbag
chair
couch
dining table
sink
potted plant
car
dog
donut
vase
remote
tv



### Quickdraw relevant objects

apple
banana
sandwich
wine bottle
coffee cup
fork
spoon
knife
wine glass
cell phone
book
clock
keyboard
laptop
mouse
scissors
backpack
purse
chair
couch
table
sink
house plant
car
dog
donut
vase
remote control
tv

