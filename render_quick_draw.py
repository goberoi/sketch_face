import cv2
import numpy as np
import pprint
import json
import random

# Settings
path = "./data/"
height = 1000
width = 1000
debug = False
image_width = 256
image_height = 180

# Helper to load data
def load_images(filename):
    images = []
    with open(path + filename) as file:
        for line in file:
            image = json.loads(line)
            images.append(image)
    return images

def draw_image(canvas, x, y, image, scale=1):
    if debug:
        cv2.circle(canvas, (int(x), int(y)), 2, (0,0,255))
        cv2.rectangle(canvas, (int(x - image_width/2), int(y - image_height/2)), (int(x + image_width/2), int(y + image_height/2)), (0,0,255), 1)

    drawing = image['drawing']
    for stroke in drawing:
        points = np.array(list(zip(stroke[0], stroke[1])))
        points = (points * scale).astype(int)
        points += [int(x - image_width * scale / 2), int(y - image_height * scale / 2)]
        cv2.polylines(canvas, [points], False, (156, 156, 156), 2)


# Initialize variables
pp = pprint.PrettyPrinter(indent=4)
images = {
    'mouth' : [],
    'nose' : [],
    'eye' : []
}

# Load images
images['mouth'] = load_images('mouth.ndjson')
images['eye'] = load_images('eye.ndjson')
images['nose'] = load_images('nose.ndjson')

# Create blank canvas
canvas = np.zeros((height,width,3), np.uint8)

while(True):
    # Clear canvas
    canvas[:, :, :] = (255, 255, 255)

    # Draw mouth
    draw_image(canvas, 
               width * 0.5, 
               height * 0.67, 
               random.choice(images['mouth']))

    # Draw left eye
    draw_image(canvas, 
               width * 0.33, 
               height * 0.33, 
               random.choice(images['eye']),
               0.5)

    # Draw right eye
    draw_image(canvas, 
               width * 0.67, 
               height * 0.33, 
               random.choice(images['eye']),
               0.5)

    # Draw nose
    draw_image(canvas, 
               width * 0.5, 
               height * 0.43, 
               random.choice(images['nose']),
               0.6)

    # Show image and wait for key
    cv2.imshow('Picture', canvas)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
