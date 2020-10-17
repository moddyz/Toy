#include <toy/application/viewportWindow.h>
#include <toy/imaging/convert.h>
#include <toy/imaging/dollyManipulator.h>
#include <toy/imaging/orbitManipulator.h>
#include <toy/imaging/truckManipulator.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

TOY_NS_OPEN

ViewportWindow::ViewportWindow( const char* i_title, const gm::Vec2i& i_dimensions )
    : toy::Window( i_title, i_dimensions )
    , m_cameraTransform( /* origin */ gm::Vec3f( 0, 0, -1 ),
                         /* target */ gm::Vec3f( 0, 0, 0 ),
                         /* up */ gm::Vec3f( 0, 1, 0 ) )
    , m_cameraView( /* verticalFov */ 90.0f,
                    /* aspectRatio */ ( float ) i_dimensions.X() / float( i_dimensions.Y() ) )
{
}

void ViewportWindow::WriteFrame( uint32_t* o_frameData )
{
    Render( m_image );
    ConvertImageVec3fToUint32( m_image, m_texture );
    CUDA_CHECK( cudaMemcpy( o_frameData, m_texture.GetBuffer(), m_texture.GetByteSize(), cudaMemcpyHostToDevice ) );
}

void ViewportWindow::OnResize( const gm::Vec2i& i_dimensions )
{
    m_image.Resize( i_dimensions.Y(), i_dimensions.X() );
    m_texture.Resize( i_dimensions.Y(), i_dimensions.X() );
}

void ViewportWindow::OnMouseMove( const gm::Vec2f& i_position )
{
    gm::Vec2f mouseDelta = i_position - GetLastMousePosition();

    if ( GetMouseButtonPressed() & toy::MouseButton_Left )
    {
        toy::OrbitManipulator orbitManip( m_cameraTransform );
        orbitManip( mouseDelta );
    }
    else if ( GetMouseButtonPressed() & toy::MouseButton_Middle )
    {
        toy::TruckManipulator truckManip( m_cameraTransform, /* sensitivity */ 0.01f );
        truckManip( mouseDelta );
    }
    else if ( GetMouseButtonPressed() & toy::MouseButton_Right )
    {
        toy::DollyManipulator dollyManip( m_cameraTransform, /* sensitivity */ 0.01f );
        dollyManip( mouseDelta.Y() );
    }
}

void ViewportWindow::OnScroll( const gm::Vec2f& i_offset )
{
    toy::DollyManipulator dollyManip( m_cameraTransform );
    dollyManip( i_offset.Y() );
}

TOY_NS_CLOSE
