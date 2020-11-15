# Notes

Random notes and references used for this project.

References:
- https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-Software-Rasterization/laine2011hpg_paper.pdf

# A trip through the graphics pipeline notes

URL: https://alain-galvan.gitbook.io/a-trip-through-the-graphics-pipeline

## The Software Stack

- Application code (ie. Quake)
- API Runtime (ie. OpenGL, DirectX)
- User mode graphics driver aka **UMD**.  Specific to graphics vendor.  Just a shared library.
- Kernel mode graphics driver (KMD)
- Bus

### API Runtime

One distinction between D3D and OpenGL:
- DirectX API compiles shader code and passes verified bytecode down to UMD
- OpenGL API does not, thus and it is the UMD's responsibility (resulting in differences bewteen implementations)

### UMD

- Where shader compilation happens.
- Multiple variants of the same shader for different API params.
- Handling of legacy shader versions.
- Writes to command or DMA buffers (allocated by KMD)

The GPU scheduler component arbitrates access to the 3D pipeline by time-slicing it between various apps needing to use the GPU.

### KMD

Deals with things that are just there once.
- GPU memory
- Allocate and map physical memory
- Hardware mouse cursor
- DRM / content protection
- Manages command buffer

### The Bus

Transfer data to GPU

## GPU memory architecture and command processor

### Memory subsystem

- GPU dram - high memory bandwidth, but also high latency.
- Memory organized into a grid (rows & cols)
- A single access fetches an entire row.
- DMA Engine (I think this is the copy engine in cuda properties?) for copying memory between system and GPU without involving shader cores.

### Command processor

- Command processing front end parses commands.
- Commands include 2D functionality, or 3D primitive handoff
- Some commands change state - which can result in a non-trivial amount of work by the hardware.
- Make a stage stateless by passing in the data along with the stage (but can be expensive for large data sets)

### Synchronization

- If Event X happens, do Y
- Can be push (GPU tells CPU it's entering vertical blank interval)
- Or pull (CPU asks GPU what was the latest parsed command buffer)
- ..

## 3D Pipeline Overview

### Terminology

- IA (Input Assembler): Reads vertex arrays (P, N, other user data) and indices array
- VS (Vertex shader): Reads input vertex data, and writes out processed vertex data.
- PA (Primitive Assembler) Reads vertices and constructs "primitives" per set of vertices.
- HS (Hull Shader) Accepts patch primitives and writes transformed patch control points.  

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

