#pragma once

/// \file memory/array.h
///
/// Core array class, abstracting memory allocation

#include <toy/utils/diagnostic.h>
#include <toy/memory/residency.h>

#include <algorithm>

TOY_NS_OPEN

/// \class Array
///
/// Array interface, for abstracting away memory allocation details of a specified device.
template < typename ValueT, Residency ResidencyT >
class Array final
{
public:
    /// \typedfe AllocatorT
    using AllocatorT = typename _GetAllocator< ResidencyT >::AllocatorT;

    //-------------------------------------------------------------------------
    /// \name Construction
    //-------------------------------------------------------------------------

    /// Default constructor of an empty array.
    ///
    /// The underlying buffer pointer will be initialized to \p nullptr.
    ///
    /// \sa GetBuffer
    Array() = default;

    /// Initialize this array with a specified size.
    ///
    /// \param i_size The size to initialize this array to.
    explicit Array( size_t i_size )
    {
        TOY_VERIFY( Resize( i_size ) );
    }

    /// Deconstructor.
    ///
    /// The underlying buffer is deallocated if the array size is non-zero.
    ~Array()
    {
        if ( m_buffer != nullptr )
        {
            TOY_VERIFY( AllocatorT::Deallocate( m_buffer ) );
        }
    }

    /// Copy constructor.
    Array( const Array< ValueT, ResidencyT >& i_array )
    {
        TOY_VERIFY( _Copy( i_array ) );
    }

    /// Copy assignment operator.
    Array< ValueT, ResidencyT >& operator=( const Array< ValueT, ResidencyT >& i_array )
    {
        TOY_VERIFY( _Copy( i_array ) );
        return *this;
    }

    //-------------------------------------------------------------------------
    /// \name Size
    //-------------------------------------------------------------------------

    /// Get the size or number of elements in the array.
    ///
    /// \return The array size.
    inline size_t GetSize() const
    {
        return m_size;
    }

    /// Check if the array is empty.
    ///
    /// This is equivalent to GetSize() == 0.
    inline bool IsEmpty() const
    {
        return GetSize() == 0;
    }

    /// Update the size of this array.
    ///
    /// \param i_size The size to resize the array to.
    ///
    /// \return Success state of this operation.
    inline bool Resize( size_t i_size )
    {
        if ( m_size == i_size )
        {
            return true;
        }

        // Resizing to 0 is a special scenario.  The buffer is de-allocated and set to nullptr.
        if ( i_size == 0 )
        {
            TOY_ASSERT( m_buffer != nullptr );
            TOY_VERIFY( AllocatorT::Deallocate( m_buffer ) );
            m_buffer = nullptr;
            m_size   = 0;
            return true;
        }

        // Try an allocate a new buffer.
        void* newBuffer = AllocatorT::Allocate( i_size * sizeof( ValueT ) );
        if ( newBuffer == nullptr )
        {
            return false;
        }

        // If there is an existing buffer, perform data migration.
        if ( m_buffer != nullptr )
        {
            size_t elementsToCopy = std::min( m_size, i_size );
            TOY_VERIFY( AllocatorT::Copy( /* dst */ newBuffer,
                                          /* src */ m_buffer,
                                          /* numBytes */ elementsToCopy * sizeof( ValueT ) ) );
            TOY_VERIFY( AllocatorT::Deallocate( m_buffer ) );
        }

        // Assign new buffer ptr & size.
        m_buffer = newBuffer;
        m_size   = i_size;

        return true;
    }

    /// Empty the array, by clearing the underlying memory.
    inline void Clear()
    {
        TOY_VERIFY( Resize( 0 ) );
    }

    //-------------------------------------------------------------------------
    /// \name Buffer access
    //-------------------------------------------------------------------------

    /// Get the underlying buffer pointer to the array.
    ///
    /// If the array is empty, then the returned value is \p nullptr.
    ///
    /// \return The underlying buffer pointer.
    inline ValueT* GetBuffer() const
    {
        TOY_ASSERT_MSG( m_buffer != nullptr, "Attempted to get null buffer pointer.\n" );
        return static_cast< ValueT* >( m_buffer );
    }

private:
    // Helper method to copy the attributes and data from a source array into this array.
    bool _Copy( const Array< ValueT, ResidencyT >& i_array )
    {
        if ( !i_array.IsEmpty() )
        {
            if ( !Resize( i_array.GetSize() ) )
            {
                return false;
            }

            return AllocatorT::Copy( /* dst */ m_buffer,
                                     /* src */ i_array.GetBuffer(),
                                     /* numBytes */ m_size * sizeof( ValueT ) );
        }
        else
        {
            return true;
        }
    }

    size_t m_size   = 0;
    void*  m_buffer = nullptr;
};

TOY_NS_CLOSE
