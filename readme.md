# Sketch Face

Use your webcam to control a cartoon version of your face. Open your
mouth to hurl cartoon objects in the direction you are facing.  Do it
alone, or enjoy hours of fun with the whole family.

![Alt Text](https://github.com/goberoi/face_experiments/blob/master/face_experiments.gif)

From gfycat:
![Alt Text](https://gfycat.com/ifr/SoreDevotedChameleon)

## Overview

I built this project as a learning excercise to understand how to work with video and images containing faces.

Key ingredients include:
* face_recognition: an excellent (which uses Dlib under the hood), OpenCV, and Google's Quickdraw dataset to produce these results. 
I also relied on lots of quality advice across the internet (see References).

## Installation

This works great on Ubuntu, but not on MacOS because 
[Docker for Mac does not pass through USB devices to containers](https://docs.docker.com/docker-for-mac/faqs/#can-i-pass-through-a-usb-device-to-a-container) (boo!).
It might work on Windows but has not been tested there.

```
git clone git@github.com:goberoi/face_experiments.git
docker build goberoi/face_experiments .
docker -ti run goberoi/face_experiments
```

## References

* This project was inspired by [this video on Instagram](https://www.instagram.com/p/BUU8TuQD6_v).
* The face_recognition library is awesome. [Start with this demo code](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py).
* On how to [increase FPS by putting camera reading IO in another thread](https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/).
* I 'borrowed' [head pose estimation code from here](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/).
