#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=models/wrn/wrn_28_10_solver.prototxt $@
