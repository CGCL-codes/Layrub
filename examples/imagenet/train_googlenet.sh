#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models/bvlc_googlenet/quick_solver.prototxt $@
