#!/usr/bin/env bash

# create a custom docker image based on:
# - tensorflow:2.4.0-gpu-jupyter and CUDA 11
# - misc packages like matplotlib, scikit-learn, tqdm

docker build -t cityu_knot_classifier .
