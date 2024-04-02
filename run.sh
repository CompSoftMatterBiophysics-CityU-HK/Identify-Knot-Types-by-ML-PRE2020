#!/usr/bin/env bash

# -v with mounted volume (current dir) for jupyter dir (/tf)
# run as user so mounted volume does not have root user files

# first check if the gpu is available
docker run --rm \
    -u $(id -u):$(id -g) \
    --gpus all \
    -it \
    -v ./:/tf/ \
    cityu_knot_classifier \
    python3 /tf/gpu.py

# launch the jupyter on current dir

docker run --rm \
    -u $(id -u):$(id -g) \
    --gpus all \
    -it \
    -v ./:/tf/ \
    -p 8888:8888 \
    cityu_knot_classifier
