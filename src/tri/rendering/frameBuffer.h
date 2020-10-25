#pragma once

/// \file rendering/frameBuffer.h
///
/// FrameBuffer class.

#include <tri/base/diagnostic.h>
#include <tri/memory/array.h>

#include <gm/types/vec3iRange.h>

#include <algorithm>

TRI_NS_OPEN

/// \class FrameBuffer
///
/// 3D dimensional array class, with templated value type and memory residency.
///
/// \tparam ValueT Type of the elements in this array.
/// \tparam ResidencyT Where the memory resides.
///
/// \pre ValueT Must be default constructable.
template < typename ValueT, Residency ResidencyT >
class FrameBuffer final
{
public:
    //-------------------------------------------------------------------------
    /// \name Construction
    //-------------------------------------------------------------------------

    /// Default constructor of an empty frameBuffer.
    FrameBuffer() = default;

    /// Initialize this frameBuffer with a specified size.
    ///
    /// \param i_width Number of rows in this frameBuffer.
    /// \param i_height Number of columns in this frameBuffer.
    explicit FrameBuffer( const gm::Vec3i& i_dims )
    {
        Resize( i_dims );
    }

    /// Initialize this array with a specified size, with an initialized value for all the
    /// elements.
    ///
    /// \param i_width Number of rows in this frameBuffer.
    /// \param i_height Number of columns in this frameBuffer.
    /// \param i_value Value to set for the elements of this frameBuffer.
    explicit FrameBuffer( const gm::Vec3i& i_dims, const ValueT& i_value )
    {
        Resize( i_dims, i_value );
    }

    /// Homogenous residency copy constructor.
    FrameBuffer( const FrameBuffer< ValueT, ResidencyT >& i_frameBuffer )
    {
        TRI_VERIFY( _Copy( i_frameBuffer ) );
    }

    /// Heterogenous residency copy constructor.
    template < Residency SrcResidencyT >
    FrameBuffer( const FrameBuffer< ValueT, SrcResidencyT >& i_frameBuffer )
    {
        TRI_VERIFY( _Copy( i_frameBuffer ) );
    }

    /// Homogenous copy assignment operator.
    FrameBuffer< ValueT, ResidencyT >& operator=( const FrameBuffer< ValueT, ResidencyT >& i_frameBuffer )
    {
        TRI_VERIFY( _Copy( i_frameBuffer ) );
        return *this;
    }

    /// Heterogenous copy assignment operator.
    template < Residency SrcResidencyT >
    FrameBuffer< ValueT, ResidencyT >& operator=( const FrameBuffer< ValueT, SrcResidencyT >& i_frameBuffer )
    {
        TRI_VERIFY( _Copy( i_frameBuffer ) );
        return *this;
    }

    //-------------------------------------------------------------------------
    /// \name Element access
    //-------------------------------------------------------------------------

    /// Indexed element access.
    inline const ValueT& operator[]( size_t i_index ) const
    {
        return m_array[ i_index ];
    }

    /// Indexed element access.
    inline ValueT& operator[]( size_t i_index )
    {
        return m_array[ i_index ];
    }

    /// 3D coordinate indexed element access.
    inline const ValueT& operator()( int i_x, int i_y, int i_z ) const
    {
        TRI_ASSERT( i_x < m_dims.X() );
        TRI_ASSERT( i_y < m_dims.Y() );
        TRI_ASSERT( i_z < m_dims.Z() );
        return m_array[ ( i_z * m_dims.X() * m_dims.Y() ) + ( i_y * m_dims.X() ) + i_x ];
    }

    /// 3D coordinate indexed element access.
    inline ValueT& operator()( int i_x, int i_y, int i_z )
    {
        TRI_ASSERT( i_x < m_dims.X() );
        TRI_ASSERT( i_y < m_dims.Y() );
        TRI_ASSERT( i_z < m_dims.Z() );
        return m_array[ ( i_z * m_dims.X() * m_dims.Y() ) + ( i_y * m_dims.X() ) + i_x ];
    }

    /// 3D coordinate indexed element access.
    inline const ValueT& operator()( const gm::Vec3i& i_coord ) const
    {
        TRI_ASSERT( i_coord.X() < m_dims.X() );
        TRI_ASSERT( i_coord.Y() < m_dims.Y() );
        TRI_ASSERT( i_coord.Z() < m_dims.Z() );
        return m_array[ ( i_coord.Z() * m_dims.X() * m_dims.Y() ) + ( i_coord.Y() * m_dims.X() ) + i_coord.X() ];
    }

    /// 3D coordinate indexed element access.
    inline ValueT& operator()( const gm::Vec3i& i_coord )
    {
        TRI_ASSERT( i_coord.X() < m_dims.X() );
        TRI_ASSERT( i_coord.Y() < m_dims.Y() );
        TRI_ASSERT( i_coord.Z() < m_dims.Z() );
        return m_array[ ( i_coord.Z() * m_dims.X() * m_dims.Y() ) + ( i_coord.Y() * m_dims.X() ) + i_coord.X() ];
    }

    //-------------------------------------------------------------------------
    /// \name Operations
    //-------------------------------------------------------------------------

    /// Get the total number of elements in this frameBuffer.
    ///
    /// \return The array size.
    inline size_t GetElementCount() const
    {
        return m_dims.X() * m_dims.Y() * m_dims.Z();
    }

    /// Get the number of bytes allocated for this frameBuffer.
    ///
    /// \return The number of bytes.
    inline size_t GetByteSize() const
    {
        return GetElementCount() * sizeof( ValueT );
    }

    /// Get the extent of this frame buffer.
    inline gm::Vec3iRange GetExtent() const
    {
        return gm::Vec3iRange( gm::Vec3i( 0, 0, 0 ), gm::Vec3i( GetWidth(), GetHeight(), GetDepth() ) );
    }

    /// Get the width of this frame buffer.
    inline size_t GetWidth() const
    {
        return m_dims.X();
    }

    /// Get the height of this frame buffer.
    inline size_t GetHeight() const
    {
        return m_dims.Y();
    }

    /// Get the depth of this frame buffer.
    inline size_t GetDepth() const
    {
        return m_dims.Z();
    }

    /// Get the dimensiosn fo this frame buffer.
    inline const gm::Vec3i& GetDimensions() const
    {
        return m_dims;
    }

    /// Check if the frameBuffer is empty.
    ///
    /// This is equivalent to GetElementCount() == 0.
    inline bool IsEmpty() const
    {
        return GetElementCount() == 0;
    }

    /// Resize this frame buffer.
    ///
    /// \param i_dims New columns size.
    ///
    /// \return Success state of this operation.
    inline bool Resize( const gm::Vec3i& i_dims )
    {
        if ( m_array.Resize( i_dims.X() * i_dims.Y() * i_dims.Z() ) )
        {
            m_dims = i_dims;
            return true;
        }
        else
        {
            return false;
        }
    }

    /// Resize this array, with initialized value.
    ///
    /// \param i_newSize The size to resize the array to.
    ///
    /// \return Success state of this operation.
    inline bool Resize( const gm::Vec3i& i_dims, const ValueT& i_value )
    {
        if ( m_array.Resize( i_dims.X() * i_dims.Y() * i_dims.Z(), i_value ) )
        {
            m_dims = i_dims;
            return true;
        }
        else
        {
            return false;
        }
    }

    /// Empty the frameBuffer, removing the underlying buffer.
    inline void Clear()
    {
        m_array.Clear();
        m_dims = gm::Vec3i();
    }

    //-------------------------------------------------------------------------
    /// \name Array and buffer access
    //-------------------------------------------------------------------------

    /// Get the underlying array.
    ///
    /// \return The underlying array.
    inline const Array< ValueT, ResidencyT >& GetArray() const
    {
        return m_array;
    }

    /// Get the underlying array.
    ///
    /// \return The underlying array.
    inline Array< ValueT, ResidencyT >& GetArray()
    {
        return m_array;
    }

    /// Get the underlying buffer pointer to the frameBuffer.
    ///
    /// If the frameBuffer is empty, then the returned value is \p nullptr.
    ///
    /// \sa IsEmpty()
    ///
    /// \return The underlying buffer pointer.
    inline ValueT* GetBuffer() const
    {
        return m_array.GetBuffer();
    }

private:
    // Helper method to copy the attributes and data from a source array into this array.
    template < Residency SrcResidencyT >
    inline bool _Copy( const FrameBuffer< ValueT, SrcResidencyT >& i_frameBuffer )
    {
        if ( m_array.Copy( i_frameBuffer.GetArray() ) )
        {
            m_dims = i_frameBuffer.GetDimensions();
            return true;
        }
        else
        {
            return false;
        }
    }

    gm::Vec3i                   m_dims;
    Array< ValueT, ResidencyT > m_array;
};

TRI_NS_CLOSE
