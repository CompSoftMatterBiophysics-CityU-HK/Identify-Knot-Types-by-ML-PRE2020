#!/usr/bin/env bash

# create a custom docker image based on:
# - tensorflow:2.4.0-gpu-jupyter and CUDA 11
# - misc packages like matplotlib and scikit-learn
docker build -t cityu_knot_classifier .

# NOTE: the data are mounted in the next step bash run.sh
# data mounted are:
# - knot data folders
# - the best trained model+weights for inference
