#pragma once

/// \file tri/tri.h
///
/// Tri API
///
/// \b Tri is a software renderer implementing a fixed rasterization pipeline.
///
/// This header provides all the API entry points of the Tri renderer.

/// \enum TriRuntimeTarget
///
/// The runtime target used to execute the graphics operations.
enum class TriRuntimeTarget : char
{
    CPU = 0,
    CUDA,
    Count
};

/// \class TriContext
///
/// Describes root-level properties of a Tri renderer.
///
/// After the context is initialized with specified properties, those
/// properties are \em immutable for the lifetime of the context object.
///
/// The client must construct a new context if different property
/// specifications are desired.
///
/// Keep a TriContext object alive for the duration produced child objects.
/// Any class which takes a TriContext as a constructor argument is
/// considered a child object of the associated TriContext object.
///
/// Example usage:
/// \code{.cpp}
/// // Expected behavior when running on a machine with CUDA runtime support.
///
/// // Preferred context.
/// TriContext preferredCtx;
/// assert(preferredCtx.GetRuntimeTarget() == TriRuntimeTarget::CUDA);
///
/// // Construct a context with CPU runtime target selection.
/// TriContext cpuCtx(TriRuntimeTarget::CPU);
/// assert(cudaCtx.GetRuntimeTarget() == TriRuntimeTarget::CPU);
/// \endcode
class TriContext : final
{
public:
    /// Construct a TriContext object with the preferred runtime target.
    ///
    /// The "preferred" runtime target is selected based on priority and
    /// availability, by starting from the largest value of \ref
    /// TriRuntimeTarget (bar TriRuntimeTarget::Count) and incrementing backwards.
    TriContext();

    /// Construct a TriContext for a requested runtime target.
    ///
    /// \note The requested runtime target may not be available, in which the
    /// fallback CPU runtime target will be selected.
    ///
    /// \param i_runtime target The runtime target used to execute graphics
    /// commands.
    explicit TriContext(TriRuntimeTarget i_requestedRuntimeTarget);

    /// Get the actual runtime target selected by this context.
    ///
    /// \return The runtime target used to execute graphics commands.
    inline TriRuntimeTarget GetRuntimeTarget() const { return m_runtimeTarget; }

private:
    TriRuntimeTarget m_runtimeTarget;
};

/// \class TriProgram
///
/// An executable program
class TriProgram : final
{
public:
    /// Construct a program which executes code for a given TriContext.
    ///
    /// \param i_context The context object providing root-level specification.
    TriProgram(const TriContext& i_context);

private:
    const TriContext& m_context;
};

/// \class TriIndexedTriangleList
///
/// The geometry specification of an object to draw in the scene.
class TriIndexedTriangleList : final
{};

/// \class TriDrawCommand
///
/// Combination of geometry, transformation parameters, and shader programs
/// which are compose a single draw invocation.
class TriDrawCall : final
{
public:
    /// Set an executable program to be run on each input vertex before primitive assembly.
    ///
    /// \param i_program The vertex program.
    SetVertexProgram(const TriProgram* i_program);
    SetFragmentProgram(const TriProgram* i_program);

private:
    const TriProgram* m_vertexProgram = nullptr;
    const TriProgram* m_fragmentProgram = nullptr;
};

/// \class TriDrawParams
///
/// Root-level parameters for drawing.
class TriDrawParams : final
{
public:
    /// Set the transformation matrix which moves an object from camera-space
    /// into world-space.
    ///
    /// Also known as the 'camera' transform.
    ///
    /// \param i_cameraTransform The camera-to-world transformation matrix.
    ///
    /// \return If the transformation was successfully set (the matrix is
    /// invertable).
    bool SetCameraTransform(const gm::Mat4f& i_cameraTransform);

    /// Set the transformation matrix which moves objects from camera space
    /// into the view-frustum or clipping space.
    ///
    /// Also known as the 'projection' transformation.
    ///
    /// \param i_projectionTransform The camera-to-clip transformation matrix.
    void SetProjectionTransform(const gm::Mat4f& i_projectionTransform);

    /// Set the transform which moves objects from clip space into viewport
    /// space.
    ///
    /// Also known as the "viewport" transformation.
    ///
    /// \param i_viewportTransform The camera-to-clip transformation matrix.
    void SetViewportTransform(const gm::Mat4f& i_viewportTransform);

private:
    gm::Mat4f m_cameraTransform;
    gm::Mat4f m_projectionTransform;
    gm::Mat4f m_viewportTransform;
};

