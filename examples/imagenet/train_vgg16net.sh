#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models/vgg/vgg16_solver.prototxt $@
