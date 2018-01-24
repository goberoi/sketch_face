# Sketch Face

Use your webcam to control a cartoon version of your face. Open your
mouth to hurl cartoon objects in the direction you are facing.  Do it
alone, or enjoy hours of fun with the whole family.

![Alt Text](https://github.com/goberoi/sketch_face/blob/master/sketch_face.gif)

## Overview


I built this project as a learning excercise to understand how to work with video and images containing faces.

Key ingredients include:
* [face_recognition](https://github.com/ageitgey/face_recognition): an excellent Python library for all things faces. Built on top of the powerful [dlib](https://github.com/davisking/dlib) toolkit.
* [OpenCV](https://github.com/opencv/opencv): a vision programmer's best friend for working with images and video.
* [Google's Quickdraw dataset](https://github.com/googlecreativelab/quickdraw-dataset): the source of images used for the eyes, nose, and mouth-spewed objects.

## Installation

The provided Dockerfile makes installation easy:

```
git clone https://github.com/goberoi/sketch_face.git
docker build -t goberoi/sketch_face .
docker -ti run goberoi/sketch_face
```

The above has been tested on Ubuntu 16.04, but not Windows, or Mac. 
Mac users, warning: [Docker for Mac does not pass through USB devices to containers](https://docs.docker.com/docker-for-mac/faqs/#can-i-pass-through-a-usb-device-to-a-container) so you'll have to find another way (workarounds do exist).

## Usage

While the program is running, click to give it focus, and then press these keys for functionality:
* 'v': toggle video background, vs. white background.
* 'p': toggle showing head pose estimate as a blue line, as well as the 6 face features used to compute it.
* 's': render eyes and nose as a cartoon sketch, or not.
* 'q': quit and cleanup.

Type `python game_face.py` to see command line options for the same as above.

## How it Works

In a nutshell:
1. In a loop, read frames from the camera, scale them down for speed, and process them as follows:
2. Detect points on a face by calling face_recognition.get_landmarks. This returns 68 points categorized by left_eye, nose_tip, etc.
3. Find the centers of the eyes and nose, and replace them with random Quickdraw cartoons of eyes and noses. Render the rest as lines.
4. If the mouth appears to be open (crudely determined by looking at the ratio of height to width), then estimate head pose using the given landmarks, and create random Quickdraw sprites moving in that direction.
5. Check for key presses and toggle various rendering options as described above.

## References

I relied on lots of quality advice and code to build this:
* This [video on Instagram](https://www.instagram.com/p/BUU8TuQD6_v) inspired the project idea.
* The face_recognition library's [demo code](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py) is a great starting point.
* Very useful info on how to [increase FPS by putting camera reading IO in another thread](https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/).
* [Head pose estimation code](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/) that I shamelessly 'borrowed'.
