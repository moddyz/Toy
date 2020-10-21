<a href="https://github.com/moddyz/Toy/actions?query=workflow%3A%22Build+and+test%22"><img src="https://github.com/moddyz/Toy/workflows/Build%20and%20test/badge.svg"/></a>

# Toy

Building blocks for producing images.

## Table of Contents

- [Dependencies](#dependencies)
- [Building](#building)
- [Code structure](#code-structure)

### Dependencies

The following dependencies are mandatory:
- C++ compiler (>=C++11)
- [CMake](https://cmake.org/documentation/) (>=3.12)
- [GLFW](https://www.glfw.org/) 
- [GLEW](http://glew.sourceforge.net/) 
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (>=10)

The following dependencies are optional:
- [Doxygen](https://www.doxygen.nl/index.html) and [graphiviz](https://graphviz.org/) for documentation.

## Building

Example snippet for building & installing this project:
```
mkdir build && cd build
cmake \
  -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" \
  -DBUILD_TESTING=ON \
  -DCMAKE_INSTALL_PREFIX="/apps/Toy/" \
  .. 
cmake --build  . -- VERBOSE=1 -j8 all test install
```
CMake options for configuring this project:

| CMake Variable name     | Description                                                            | Default |
| ----------------------- | ---------------------------------------------------------------------- | ------- |
| `CMAKE_CUDA_COMPILER`   | Path to the nvcc executable.                                           | `OFF`   |
| `BUILD_TESTING`         | Enable automated testing.                                              | `OFF`   |
| `BUILD_DOCUMENTATION`   | Build documentation.                                                   | `OFF`   |

## Code structure

[memory](./src/toy/memory) provides the custom heterogenous (CPU/CUDA) `Array` and `Matrix` (2D array) classes commonly used in image making.

The base [imaging](./src/toy/imaging) library provides common facilities used in both [rasterization](./src/toy/rasteriation) and [raytracing](./src/toy/raytracing) techniques.

[viewport](./src/toy/viewport) provides viewport back-end and user interactions, and finally, [application](./src/toy/application) provides tools for windowing.
