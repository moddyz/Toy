#!/bin/bash

set -euxo pipefail

mkdir -p build && cd build

CMAKE_ARGS=\
\ -DCMAKE_BUILD_TYPE="Debug"\
\ -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc"\
\ -DBUILD_DOCUMENTATION="TRUE"\
\ -DBUILD_TESTING="TRUE"

# Only build if installation path not specified.
if [ $# -eq 0 ]
then
    cmake $CMAKE_ARGS ..
    cmake --build . -- all test -j 8
else
    cmake $CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=$1 ..
    cmake --build . --target install -- VERBOSE=1 -j 8
fi
