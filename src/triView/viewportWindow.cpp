#include "viewportWindow.h"
#include "dollyManipulator.h"
#include "orbitManipulator.h"
#include "truckManipulator.h"

#include <tri/rendering/formatConversion.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

ViewportWindow::ViewportWindow(const char* i_title,
                               const gm::Vec2i& i_dimensions)
  : tri::Window(i_title, i_dimensions)
  , m_cameraTransform(/* origin */ gm::Vec3f(0, 0, -1),
                      /* target */ gm::Vec3f(0, 0, 0),
                      /* up */ gm::Vec3f(0, 1, 0))
  , m_cameraView(/* verticalFov */ 90.0f,
                 /* aspectRatio */ (float)i_dimensions.X() /
                     float(i_dimensions.Y()))
{}

void
ViewportWindow::WriteFrame(uint32_t* o_frameBuffer)
{
    Render(m_colorBuffer);
    ConvertRGBFloatToRGBAUint32<CUDA>::Execute(m_colorBuffer.GetElementCount(),
                                               m_colorBuffer.GetBuffer(),
                                               o_frameBuffer);
}

void
ViewportWindow::OnResize(const gm::Vec2i& i_dimensions)
{
    m_colorBuffer.Resize(gm::Vec3i(i_dimensions.X(), i_dimensions.Y(), 1));
}

void
ViewportWindow::OnMouseMove(const gm::Vec2f& i_position)
{
    gm::Vec2f mouseDelta = i_position - GetLastMousePosition();

    if (GetMouseButtonPressed() & tri::MouseButton_Left) {
        tri::OrbitManipulator orbitManip(m_cameraTransform);
        orbitManip(mouseDelta);
    } else if (GetMouseButtonPressed() & tri::MouseButton_Middle) {
        tri::TruckManipulator truckManip(m_cameraTransform,
                                         /* sensitivity */ 0.01f);
        truckManip(mouseDelta);
    } else if (GetMouseButtonPressed() & tri::MouseButton_Right) {
        tri::DollyManipulator dollyManip(m_cameraTransform,
                                         /* sensitivity */ 0.01f);
        dollyManip(mouseDelta.Y());
    }
}

void
ViewportWindow::OnScroll(const gm::Vec2f& i_offset)
{
    tri::DollyManipulator dollyManip(m_cameraTransform);
    dollyManip(i_offset.Y());
}
