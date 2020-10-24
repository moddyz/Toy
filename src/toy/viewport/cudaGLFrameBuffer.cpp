#include <toy/base/diagnostic.h>
#include <toy/base/log.h>
#include <toy/memory/cudaError.h>
#include <toy/viewport/cudaGLFrameBuffer.h>

#include <GL/glew.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

TOY_NS_OPEN

CUDAGLFrameBuffer::CUDAGLFrameBuffer( int i_width, int i_height )
    : m_width( i_width )
    , m_height( i_height )
{
    TOY_LOG_DEBUG( "Creating frame buffer with dimensions (w=%i, h=%i).\n", m_width, m_height );

    //
    // 1. Allocate GL buffer.
    //

    // Initialize a pixel buffer object.
    glGenBuffers( 1, &m_pixelBufferId );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_pixelBufferId );
    glBufferData( GL_PIXEL_UNPACK_BUFFER, GetByteSize(), 0, GL_DYNAMIC_COPY );

    // Register a CUDA graphics resource with respect to the PBO.
    CUDA_CHECK(
        cudaGraphicsGLRegisterBuffer( &m_graphicsResource, m_pixelBufferId, cudaGraphicsMapFlagsWriteDiscard ) );

    //
    // 2. Allocate texture.
    //

    // Enable Texturing
    glEnable( GL_TEXTURE_2D );

    // Generate a texture ID
    glGenTextures( 1, &m_textureId );

    // Make this the current texture (remember that GL is state-based)
    glBindTexture( GL_TEXTURE_2D, m_textureId );

    // Allocate the texture memory. The last parameter is NULL since we only
    // want to allocate memory, not initialize it
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );

    // Must set the filter mode, GL_LINEAR enables interpolation when scaling
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    TOY_VERIFY( glGetError() == GL_NO_ERROR );
}

CUDAGLFrameBuffer::~CUDAGLFrameBuffer()
{
    TOY_LOG_DEBUG( "Destroying frame buffer.\n" );
    cudaGraphicsUnregisterResource( m_graphicsResource );
    glDeleteBuffers( 1, &m_pixelBufferId );
    glDeleteTextures( 1, &m_textureId );
}

uint32_t* CUDAGLFrameBuffer::WriteFrameBegin()
{
    CUDA_CHECK( cudaGraphicsMapResources( 1, &m_graphicsResource, 0 ) );
    uint32_t* devicePtr = nullptr;
    size_t    numBytes;
    CUDA_CHECK( cudaGraphicsResourceGetMappedPointer( ( void** ) &devicePtr, &numBytes, m_graphicsResource ) );
    TOY_VERIFY( numBytes == GetByteSize() );
    return devicePtr;
}

void CUDAGLFrameBuffer::WriteFrameEnd()
{
    cudaDeviceSynchronize();
    CUDA_CHECK( cudaGraphicsUnmapResources( 1, &m_graphicsResource, 0 ) );
}

void CUDAGLFrameBuffer::DrawFrame()
{
    glViewport( 0, 0, m_width, m_height );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 );

    // Select the appropriate pixel buffer.
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_pixelBufferId );

    // Select the appropriate texture.
    glBindTexture( GL_TEXTURE_2D, m_textureId );

    // Make a texture from the buffer.
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL );

    // Draw texture on a single quad.
    // The texture coordinates are flipped vertically because the drawing code
    // indexes into a 2D array, as if the (0, 0) coordinate is at the top-left corner.
    glBegin( GL_QUADS );
    glTexCoord2f( 0, -1.0f );
    glVertex3f( 0, 0, 0 );
    glTexCoord2f( 0, 0 );
    glVertex3f( 0, 1.0f, 0 );
    glTexCoord2f( 1.0f, 0 );
    glVertex3f( 1.0f, 1.0f, 0 );
    glTexCoord2f( 1.0f, -1.0f );
    glVertex3f( 1.0f, 0, 0 );
    glEnd();
}

TOY_NS_CLOSE
