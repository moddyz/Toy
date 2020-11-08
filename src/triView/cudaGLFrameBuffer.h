#pragma once

#include <stddef.h>
#include <stdint.h>

// Forward declaration.
struct cudaGraphicsResource;

/// \class CUDAGLFrameBuffer
///
/// Interface to simplify mapping and unmapping of a CUDA buffer for
/// display purposes as a OpenGL texture.
///
/// Dynamically instantiate this class for a given target display
/// dimension.
///
/// In the rendering loop call the following, in order:
/// 1. Map() which provide a frame buffer for CUDA operations to
/// write into.
/// 2. Unmap() once the CUDA computations end.
/// 3. Draw() to display the frame buffer as a texture via GL.
class CUDAGLFrameBuffer
{
public:
    //------------------------------------------------------------------------0
    /// \name Construction
    //------------------------------------------------------------------------0

    /// Initialize the CUDA and GL resources for a given dimension.
    explicit CUDAGLFrameBuffer(int i_width, int i_height);

    /// Deconstructor.
    ~CUDAGLFrameBuffer();

    //------------------------------------------------------------------------0
    /// \name Size and dimensions
    //------------------------------------------------------------------------0

    inline int GetWidth() const { return m_width; }

    inline int GetHeight() const { return m_height; }

    inline size_t GetByteSize() const { return 4 * m_width * m_height; }

    //------------------------------------------------------------------------0
    /// \name Graphics interop
    //------------------------------------------------------------------------0

    /// Begin the CUDA computation phase for writing into the frame buffer.
    ///
    /// Provides a device pointer to a frame buffer for writing into (via CUDA
    /// API).
    ///
    /// TODO: Returning a raw pointer sucks.  We can do better.
    ///
    /// \return The CUDA device pointer to a frame buffer.
    uint32_t* Map();

    /// End the CUDA computation phase for writing into a frame.
    ///
    /// \param o_frameData The CUDA device pointer to a frame buffer.
    void Unmap();

    /// Draw contents of color buffer on a OpenGL texture bound to a single
    /// quad.
    void Draw();

private:
    // Frame dimensions.
    int m_width = 0;
    int m_height = 0;

    // OpenGL members.
    uint32_t m_textureId = 0;
    uint32_t m_pixelBufferId = 0;

    // CUDA members.
    cudaGraphicsResource* m_graphicsResource = nullptr;
};
