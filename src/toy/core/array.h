#pragma once

/// \file core/array.h
///
/// Core array class, abstracting memory allocation

#include <toy/toy.h>
#include <toy/utils/diagnostic.h>

TOY_NS_OPEN

/// \class Array
///
/// Array interface, for abstracting away memory allocation details of a specified device.
template < typename ValueT, typename AllocatorT >
class Array
{
public:
    /// Default constructor of an empty array.
    ///
    /// The underlying buffer pointer is \p nullptr at this point, so be cautious
    /// when accessing this pointer though \ref GetBuffer.
    Array()
    {
    }

    /// Initialize this array with a specified size.
    ///
    /// \param i_size The size to initialize this array to.
    Array( size_t i_size )
    {
        TOY_VERIFY( Resize( i_size ) );
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
            return true;
        }

        // Allocate new buffer.
        void* newBuffer = AllocatorT::Allocate( i_size * sizeof( ValueT ) );

        // If there is an existing buffer, perform data migration.
        if ( m_buffer != nullptr )
        {
            size_t bytesToCopy = std::min( m_size, i_size );
            TOY_VERIFY( AllocatorT::Copy( /* dst */ newBuffer, /* src */ m_buffer,  ) );
            AllocatorT::Deallocate( m_buffer );
        }

        m_buffer = newBuffer;
    }

    /// Empty the array, by clearing the underlying memory.
    inline void Clear()
    {
        TOY_VERIFY( Resize( 0 ) );
    }

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
    size_t m_size   = 0;
    void*  m_buffer = nullptr;
};

TOY_NS_CLOSE
