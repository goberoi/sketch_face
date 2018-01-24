"""Interface to the Google Quickdraw dataset.

This class assumes that you have a directory containing Google
Quickdraw .ndjson files where the name of each file is the name of the
image class.

Here's the Quickdraw dataset:
https://github.com/googlecreativelab/quickdraw-dataset

"""

import cv2
import numpy as np
import json
import random
import os

# Settings
IMAGES_DIR = 'quickdraw_images'
debug = False

class QuickDraw:
    """Interface to Google Quickdraw images."""

    def __init__(self):
        self._recent_image = {}
        self._images = {}
        self._class_names = []
        for file in os.listdir(IMAGES_DIR):
            if file.endswith(".ndjson"):
                name = file.split('.')[0]
                self._images[name] = self._load_images(file)
                self._class_names.append(name)


    # Helper to load data
    @classmethod
    def _load_images(cls, filename):
        images = []
        with open(os.path.join(IMAGES_DIR, filename)) as file:
            for line in file:
                image = json.loads(line)
                images.append(image)
        return images

    def get_random(self, name=None, chance_to_pick_new = 10):
        """Return a random image of the given class name with the given
        probability to pick a new one, vs. return the last one.

        """

        if not name:
            name = random.choice(self._class_names)

        random_image = random.choice(self._images[name])
        recent_image = self._recent_image.get(name, random_image)
        if (random.randint(1,100) < chance_to_pick_new):
            self._recent_image[name] = random_image
            return random_image
        else:
            return recent_image

    @classmethod
    def render(cls, canvas, x, y, image, scale=1):
        """Render the given image on the given numpy array. Scale down using the scale factor provided."""

        image_width = 256
        image_height = 180

        if debug:
            cv2.circle(canvas, (int(x), int(y)), 2, (0,0,255))
            cv2.rectangle(canvas, (int(x - image_width/2), int(y - image_height/2)), (int(x + image_width/2), int(y + image_height/2)), (0,0,255), 1)

        drawing = image['drawing']
        for stroke in drawing:
            points = np.array(list(zip(stroke[0], stroke[1])))
            points = (points * scale).astype(int)
            points += [int(x - image_width * scale / 2), int(y - image_height * scale / 2)]
            cv2.polylines(canvas, [points], False, (156, 156, 156), 2)

if __name__ == "__main__":
    # Create blank canvas
    height = 1000
    width = 1000
    canvas = np.zeros((height,width,3), np.uint8)

    quickdraw = QuickDraw()

    while(True):
        # Clear canvas
        canvas[:, :, :] = (255, 255, 255)

        # Draw mouth
        quickdraw.render(canvas, 
                         width * 0.5, 
                         height * 0.67, 
                         quickdraw.get_random('mouth'))

        # Draw left eye
        eye = quickdraw.get_random('eye')
        quickdraw.render(canvas, 
                         width * 0.33, 
                         height * 0.33, 
                         eye,
                         0.5)

        # Draw right eye
        quickdraw.render(canvas, 
                         width * 0.67, 
                         height * 0.33, 
                         eye,
                         0.5)

        # Draw nose
        quickdraw.render(canvas, 
                         width * 0.5, 
                         height * 0.43, 
                         quickdraw.get_random('nose'),
                         0.6)

        # Show image and wait for key
        cv2.imshow('Picture', canvas)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
