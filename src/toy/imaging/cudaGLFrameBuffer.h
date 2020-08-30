#pragma once

/// \file imaging/cudaGLFrameBuffer.h
///
/// A method for enabling CUDA computations to be performed against a OpenGL-allocated
/// pixel buffer, and being able to display the pixel buffer as an image via a texture.

#include <toy/toy.h>

#include <stdint.h>
#include <stddef.h>

// Forward declaration.
struct cudaGraphicsResource;

TOY_NS_OPEN

/// \class CudaGLFrameBuffer
///
/// CudaGLFrameBuffer abstracts the process of using CUDA for producing an frame
/// to be displayed on screen via a OpenGL texture.
///
/// To use - dynamically instantiate this class for a given target display dimension.
/// If the dimensions of the target display changes - the pipeline will need to be re-created.
///
/// Then, in the rendering loop call the following, in order:
/// 1. ComputeFrameBegin() which provide a frame buffer for CUDA operations to write into.
/// 2. ComputeFrameEnd() once the CUDA computations end.
/// 3. DrawFrame() to display the frame buffer as a texture via GL.
class CudaGLFrameBuffer
{
public:
    //------------------------------------------------------------------------0
    /// \name Construction
    //------------------------------------------------------------------------0

    /// Initialize the CUDA and GL resources for a given dimension.
    explicit CudaGLFrameBuffer( int i_width, int i_height );
    ~CudaGLFrameBuffer();

    //------------------------------------------------------------------------0
    /// \name Size and dimensions
    //------------------------------------------------------------------------0

    inline int GetWidth() const
    {
        return m_width;
    }

    inline int GetHeight() const
    {
        return m_height;
    }

    inline size_t GetByteSize() const
    {
        return 4 * m_width * m_height;
    }

    //------------------------------------------------------------------------0
    /// \name Compute and graphics
    //------------------------------------------------------------------------0

    /// Begin the CUDA computation phase for writing into the frame buffer.
    ///
    /// Provides a device pointer to a frame buffer for writing into (via CUDA API).
    ///
    /// TODO: Returning a raw pointer sucks.  We can do better.
    ///
    /// \return The CUDA device pointer to a frame buffer.
    uint32_t* ComputeFrameBegin();

    /// End the CUDA computation phase for writing into a frame.
    ///
    /// \param o_frameData The CUDA device pointer to a frame buffer.
    void ComputeFrameEnd( uint32_t* o_frameData );

    /// Draw frame buffer via a OpenGL texture bound to a single quad.
    void DrawFrame();

private:
    // Frame dimensions.
    int m_width  = 0;
    int m_height = 0;

    // OpenGL members.
    uint32_t m_textureId     = 0;
    uint32_t m_pixelBufferId = 0;

    // Cuda members.
    cudaGraphicsResource* m_graphicsResource = nullptr;
};

TOY_NS_CLOSE

