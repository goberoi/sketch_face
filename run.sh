#!/bin/bash

# Allow xhost connections from any clients
xhost +

# Run docker image. Pass in usb webcam, display for output, etc.
docker run \
    -ti --rm \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/home/developer/.Xauthority \
    --net=host \
    --pid=host \
    --ipc=host \
    --device /dev/video0 \
    goberoi/sketch_face

# Disable xhost connections from any clients
xhost -
