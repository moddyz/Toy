#pragma once

/// \file memory/matrix.h
///
/// Matrix class.

#include <toy/memory/array.h>
#include <toy/utils/diagnostic.h>

#include <algorithm>

TOY_NS_OPEN

/// \class Matrix
///
/// Two-dimensional array class, with templated value type and memory residency.
///
/// \tparam ValueT Type of the elements in this array.
/// \tparam ResidencyT Where the memory resides.
///
/// \pre ValueT Must be default constructable.
template < typename ValueT, Residency ResidencyT >
class Matrix final
{
public:
    //-------------------------------------------------------------------------
    /// \name Construction
    //-------------------------------------------------------------------------

    /// Default constructor of an empty matrix.
    Matrix() = default;

    /// Initialize this matrix with a specified size.
    ///
    /// \param i_rows Number of rows in this matrix.
    /// \param i_cols Number of columns in this matrix.
    explicit Matrix( size_t i_rows, size_t i_cols )
    {
        Resize( i_rows, i_cols );
    }

    /// Initialize this array with a specified size, with an initialized value for all the
    /// elements.
    ///
    /// \param i_rows Number of rows in this matrix.
    /// \param i_cols Number of columns in this matrix.
    /// \param i_value Value to set for the elements of this matrix.
    explicit Matrix( size_t i_rows, size_t i_cols, const ValueT& i_value )
    {
        Resize( i_rows, i_cols, i_value );
    }

    /// Homogenous residency copy constructor.
    Matrix( const Matrix< ValueT, ResidencyT >& i_matrix )
    {
        TOY_VERIFY( _Copy( i_matrix ) );
    }

    /// Heterogenous residency copy constructor.
    template < Residency SrcResidencyT >
    Matrix( const Matrix< ValueT, SrcResidencyT >& i_matrix )
    {
        TOY_VERIFY( _Copy( i_matrix ) );
    }

    /// Homogenous copy assignment operator.
    Matrix< ValueT, ResidencyT >& operator=( const Matrix< ValueT, ResidencyT >& i_matrix )
    {
        TOY_VERIFY( _Copy( i_matrix ) );
        return *this;
    }

    /// Heterogenous copy assignment operator.
    template < Residency SrcResidencyT >
    Matrix< ValueT, ResidencyT >& operator=( const Matrix< ValueT, SrcResidencyT >& i_matrix )
    {
        TOY_VERIFY( _Copy( i_matrix ) );
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

    /// Row-column indexed element access.
    inline const ValueT& operator()( size_t i_row, size_t i_col ) const
    {
        TOY_ASSERT( i_row < m_rows );
        TOY_ASSERT( i_col < m_cols );
        return m_array[ i_row * m_cols + i_col ];
    }

    /// Row-column indexed element access.
    inline ValueT& operator()( size_t i_row, size_t i_col )
    {
        TOY_ASSERT( i_row < m_rows );
        TOY_ASSERT( i_col < m_cols );
        return m_array[ i_row * m_cols + i_col ];
    }

    //-------------------------------------------------------------------------
    /// \name Operations
    //-------------------------------------------------------------------------

    /// Get the total number of elements in this matrix.
    ///
    /// \return The array size.
    inline size_t GetSize() const
    {
        return m_rows * m_cols;
    }

    /// Get the height of this matrix.
    ///
    /// \return The height.
    inline size_t GetRows() const
    {
        return m_rows;
    }

    /// Get the width of this matrix.
    ///
    /// \return The width.
    inline size_t GetColumns() const
    {
        return m_cols;
    }

    /// Check if the matrix is empty.
    ///
    /// This is equivalent to GetSize() == 0.
    inline bool IsEmpty() const
    {
        return GetSize() == 0;
    }

    /// Resize this array.
    ///
    /// \param i_rows New row size.
    /// \param i_cols New columns size.
    ///
    /// \return Success state of this operation.
    inline bool Resize( size_t i_rows, size_t i_cols )
    {
        if ( m_array.Resize( i_rows * i_cols ) )
        {
            m_rows = i_rows;
            m_cols = i_cols;
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
    inline bool Resize( size_t i_rows, size_t i_cols, const ValueT& i_value )
    {
        if ( m_array.Resize( i_rows * i_cols, i_value ) )
        {
            m_rows = i_rows;
            m_cols = i_cols;
            return true;
        }
        else
        {
            return false;
        }
    }

    /// Empty the matrix, removing the underlying buffer.
    inline void Clear()
    {
        m_array.Clear();
        m_rows = 0;
        m_cols = 0;
    }

    //-------------------------------------------------------------------------
    /// \name Buffer access
    //-------------------------------------------------------------------------

    /// Get the underlying buffer pointer to the matrix.
    ///
    /// If the matrix is empty, then the returned value is \p nullptr.
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
    inline bool _Copy( const Matrix< ValueT, SrcResidencyT >& i_matrix )
    {
        if ( m_array._Copy( i_matrix.m_array ) )
        {
            m_rows = i_matrix.m_rows;
            m_cols = i_matrix.m_cols;
            return true;
        }
        else
        {
            return false;
        }
    }

    size_t                      m_rows = 0;
    size_t                      m_cols = 0;
    Array< ValueT, ResidencyT > m_array;
};

TOY_NS_CLOSE

