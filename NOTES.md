# Notes

Random notes and references used for this project.

References:
- https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-Software-Rasterization/laine2011hpg_paper.pdf

# A trip through the graphics pipeline notes

URL: https://alain-galvan.gitbook.io/a-trip-through-the-graphics-pipeline

## Layers of a 3D Application

- Application code (ie. Quake)
- API Runtime (ie. OpenGL, DirectX)
- User mode graphics driver aka **UMD**.  Specific to graphics vendor.  Just a shared library.
- Kernel mode graphics driver (KMD)
- Bus

## API Runtime

One distinction between D3D and OpenGL:
- DirectX API compiles shader code and passes verified bytecode down to UMD
- OpenGL API does not, thus and it is the UMD's responsibility (resulting in differences bewteen implementations)

## UMD

- Where shader compilation happens.
- Multiple variants of the same shader for different API params.
- Handling of legacy shader versions.
- Writes to command or DMA buffers (allocated by KMD)

The GPU scheduler component arbitrates access to the 3D pipeline by time-slicing it between various apps needing to use the GPU.

## KMD

Deals with things that are just there once.
- GPU memory
- Allocate and map physical memory
- Hardware mouse cursor
- DRM / content protection
- Manages command buffer

## The Bus

Transfer data to GPU


# CURE notes

URL: https://markussteinberger.net/papers/cuRE.pdf

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

*Object-space parallelism* - distributing work per primitive (triangle)
*Screen-space parallelism* - distributing work per screen region

The transition from object-space to screen-space is the crux of the problem.

## Design considerations

*Preservation of primitive order* - fragments must be blended in the same order in which their generating primitives were specified,

Fragment shaders must be able to compute screen-space derivatives for dependent variables, e.g., to support mipmapping

*Low memory overhead*

What does sort-first, sort-last, sort-middle, and sort-everywhere actually mean?

Tradeioff between global communication (access to global mem) and using global mem to coordinate load balancing.

## Implementation Highlights

- Vertex input indices de-duplication
- Shared memory for passing vertex shader outputs into primitive processing
- During rasterization, frame buffer is divided into bins of configurable size.
- Each bin is assigned a "rasterizer" (multiple bins can use the same rasterizer)
- Each Rasterizer is associated with a single thread block.  Thus has exclusive access to associated frame buffer bins.

Rough implementation:

Input vertex data -> Vertex Processing -> Primitive Processing -> Bin Rasterizer -> Tile rasterizer -> Fragment Processing

