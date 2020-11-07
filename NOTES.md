# Notes

Random notes and sketches for this project.

References:
- https://markussteinberger.net/papers/cuRE.pdf
- https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-Software-Rasterization/laine2011hpg_paper.pdf
- https://alain-galvan.gitbook.io/a-trip-through-the-graphics-pipeline

Multi-pass implementation for simplicity.  Results in higher intermediate memory usage and risk of a stage with low occupancy.

Streaming for lower memory overhead and higher occupancy.  Implemented using a _persistent threads_ architecture which pulls drawing items from a global work queue.
