# Notes

Random notes and sketches for this project.

References:
- https://markussteinberger.net/papers/cuRE.pdf
- https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-Software-Rasterization/laine2011hpg_paper.pdf
- https://alain-galvan.gitbook.io/a-trip-through-the-graphics-pipeline

## Outline of graphics pipeline

The graphics pipeline can be roughly outlined as follows..

Vertex processing:
- Input assembly
- Vertex shading

Primitive processing:
- Primitive assembly
- Clipping / culling
- Triangle setup

Rasterization
- Rasterizer (triangle -> fragments)
- Fragment shading
- Raster operations (2D compositing)

## Approaches

Multi-pass implementation for simplicity.  Results in higher intermediate memory usage and risk of a stage with low occupancy.

Streaming for lower memory overhead and higher occupancy.  Implemented using a _persistent threads_ architecture which pulls drawing items from a global work queue.

## Achieving parallelism

