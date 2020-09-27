# Toy

Building blocks for producing images.

The data structures and imaging algorithms are available as both CPU & CUDA implementations.

## Table of Contents

- [Code structure](#code-structure)
- [Documentation](#documentation)
- [Building](#building)
  - [Requirements](#requirements)
- [Build Status](#build-status)

## Code structure

The core [imaging](./src/toy/imaging) library provides common facilities used in both [rasterization](./src/toy/rasteriation) and [raytracing](./src/toy/raytracing) techniques.

## Documentation

Documentation based on the latest state of master, [hosted by GitHub Pages](https://moddyz.github.io/Toy/).

## Building

A convenience build script is provided, for building all targets, and optionally installing to a location:
```
./build.sh <OPTIONAL_INSTALL_LOCATION>
```

### Requirements

- `>= CMake-3.17`
- `>= C++17`
- `doxygen` and `graphviz` (optional for documentation)

## Build Status

|       | master | 
| ----- | ------ | 
| macOS-10.14 | [![Build Status](https://travis-ci.com/moddyz/Toy.svg?branch=master)](https://travis-ci.com/moddyz/Toy) |

