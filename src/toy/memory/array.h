#pragma once

/// \file memory/array.h
///
/// Core array class.

#include <toy/memory/allocate.h>
#include <toy/memory/compare.h>
#include <toy/memory/copy.h>
#include <toy/memory/deallocate.h>
#include <toy/memory/fill.h>
#include <toy/memory/index.h>
#include <toy/memory/residency.h>
#include <toy/utils/diagnostic.h>
#include <toy/utils/typeName.h>

#include <algorithm>
#include <sstream>

TOY_NS_OPEN

// Forward declarations.
template < typename ValueT, Residency ResidencyT >
class Matrix;

/// \class Array
///
/// One-dimensional array class, with templated value type and memory residency.
///
/// \tparam ValueT Type of the elements in this array.
/// \tparam ResidencyT Where the memory resides.
///
/// \pre ValueT Must be default constructable.
template < typename ValueT, Residency ResidencyT >
class Array final
{
public:
    /// \typedef ValueType
    ///
    /// Convenience type definition for this array's value type.
    using ValueType = ValueT;

    /// \typedef ResidencyType
    ///
    /// Convenience type definition for this array's residency type.
    const static Residency ResidencyType = ResidencyT;

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

    /// Initialize this array with a specified size, with an initialized value for all the
    /// elements.
    ///
    /// \param i_size The size to initialize this array to.
    explicit Array( size_t i_size, const ValueT& i_value )
    {
        TOY_VERIFY( Resize( i_size, i_value ) );
    }

    /// Initializer list constructor.
    ///
    /// \param i_initializerList The initializer list to set the array to.
    Array( std::initializer_list< ValueT > i_initializerList )
    {
        TOY_VERIFY( _CopyInitializerList( i_initializerList ) );
    }

    /// Deconstructor.
    ///
    /// The underlying buffer is deallocated if the array size is non-zero.
    ~Array()
    {
        if ( m_buffer != nullptr )
        {
            TOY_VERIFY( MemoryDeallocate< ResidencyT >::Execute( m_buffer ) );
        }
    }

    /// Homogenous residency copy constructor.
    Array( const Array< ValueT, ResidencyT >& i_array )
    {
        TOY_VERIFY( _Copy( i_array ) );
    }

    /// Heterogenous residency copy constructor.
    template < Residency SrcResidencyT >
    Array( const Array< ValueT, SrcResidencyT >& i_array )
    {
        TOY_VERIFY( _Copy( i_array ) );
    }

    /// Homogenous copy assignment operator.
    Array< ValueT, ResidencyT >& operator=( const Array< ValueT, ResidencyT >& i_array )
    {
        TOY_VERIFY( _Copy( i_array ) );
        return *this;
    }

    /// Heterogenous copy assignment operator.
    template < Residency SrcResidencyT >
    Array< ValueT, ResidencyT >& operator=( const Array< ValueT, SrcResidencyT >& i_array )
    {
        TOY_VERIFY( _Copy( i_array ) );
        return *this;
    }

    /// Initializer list copy assignment operator.
    Array< ValueT, ResidencyT >& operator=( const std::initializer_list< ValueT >& i_initializerList )
    {
        TOY_VERIFY( _CopyInitializerList( i_initializerList ) );
        return *this;
    }

    //-------------------------------------------------------------------------
    /// \name Comparison operators
    //-------------------------------------------------------------------------

    /// Check if two arrays are equivalent.
    ///
    /// An array is equivalent if it has the same size and stores the exact same
    /// element values.
    ///
    /// \param i_array The other array.
    ///
    /// \retval true If \p i_array is equivalent to this array.
    inline bool operator==( const Array< ValueT, ResidencyT >& i_array ) const
    {
        if ( GetSize() != i_array.GetSize() )
        {
            return false;
        }

        return MemoryCompare< ResidencyT >::Execute( GetSize(), GetBuffer(), i_array.GetBuffer() );
    }

    inline bool operator!=( const Array< ValueT, ResidencyT >& i_array ) const
    {
        return !( operator==( i_array ) );
    }

    //-------------------------------------------------------------------------
    /// \name Carndinality
    //-------------------------------------------------------------------------

    /// Get the size or number of elements in the array.
    ///
    /// \return The array size.
    inline size_t GetSize() const
    {
        return m_size;
    }

    /// Get the number of bytes allocated for this array.
    ///
    /// \return The number of bytes.
    inline size_t GetByteSize() const
    {
        return m_size * sizeof( ValueT );
    }

    /// Check if the array is empty.
    ///
    /// This is equivalent to GetSize() == 0.
    inline bool IsEmpty() const
    {
        return GetSize() == 0;
    }

    /// Resize this array.
    ///
    /// \param i_newSize The size to resize the array to.
    ///
    /// \return Success state of this operation.
    inline bool Resize( size_t i_newSize )
    {
        return _Resize( i_newSize );
    }

    /// Resize this array, with initialized value.
    ///
    /// \param i_newSize The size to resize the array to.
    /// \param i_value The value to fill in the expanded region from the resize.
    ///
    /// \return Success state of this operation.
    inline bool Resize( size_t i_newSize, const ValueT& i_value )
    {
        size_t oldSize = m_size;
        if ( !_Resize( i_newSize ) )
        {
            return false;
        }

        return _Fill( oldSize, m_size, i_value );
    }

    /// Empty the array, by clearing the underlying memory.
    inline void Clear()
    {
        TOY_VERIFY( Resize( 0 ) );
    }

    //-------------------------------------------------------------------------
    /// \name Element access
    //-------------------------------------------------------------------------

    inline const ValueT& operator[]( size_t i_index ) const
    {
        TOY_VERIFY( m_buffer != nullptr );
        return MemoryIndex< ResidencyT >::Execute( m_buffer, i_index );
    }

    inline ValueT& operator[]( size_t i_index )
    {
        TOY_VERIFY( m_buffer != nullptr );
        return MemoryIndex< ResidencyT >::Execute( m_buffer, i_index );
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
        return m_buffer;
    }

    //-------------------------------------------------------------------------
    /// \name Debugging
    //-------------------------------------------------------------------------

    inline std::string GetStr() const
    {
        std::stringstream ss;
        ss << "toy::Array< " << DemangledTypeName< ValueT >() << ", " << ResidencyT << " >";
        ss << "[size=" << GetSize() << "]\n";
        return ss.str();
    }

private:
    // Hmm how does array become friends with _all_ the residencies.
    friend class Matrix< ValueT, Host >;
    friend class Matrix< ValueT, Cuda >;

    // Helper method to copy the attributes and data from a source array into this array.
    template < Residency SrcResidencyT >
    inline bool _Copy( const Array< ValueT, SrcResidencyT >& i_array )
    {
        if ( !i_array.IsEmpty() )
        {
            if ( !Resize( i_array.GetSize() ) )
            {
                return false;
            }

            return MemoryCopy< SrcResidencyT, ResidencyT >::Execute(
                /* numBytes */ m_size * sizeof( ValueT ),
                /* src */ i_array.GetBuffer(),
                /* dst */ m_buffer );
        }
        else
        {
            return true;
        }
    }

    // Helper method to copy the values from the initializer list into this array.
    inline bool _CopyInitializerList( std::initializer_list< ValueT > i_initializerList )
    {
        if ( ResidencyT == Cuda )
        {
            // If _this_ array is Cuda-based, we need an extra staging host array.

            // Check for host array allocation failure (rare).
            Array< ValueT, Host > hostArray( i_initializerList.size() );
            TOY_VERIFY( hostArray.GetBuffer() != nullptr );

            // Copy values from initializer list into staging host array.
            size_t index = 0;
            for ( auto it = i_initializerList.begin(); it != i_initializerList.end(); ++it, ++index )
            {
                hostArray[ index ] = *it;
            }

            // Upload to Cuda.
            return _Copy( hostArray );
        }
        else
        {
            if ( !Resize( i_initializerList.size() ) )
            {
                return false;
            }

            size_t index = 0;
            for ( auto it = i_initializerList.begin(); it != i_initializerList.end(); ++it, ++index )
            {
                m_buffer[ index ] = *it;
            }

            return true;
        }
    }

    // Utility method for resizing this current array.
    inline bool _Resize( size_t i_newSize )
    {
        if ( m_size == i_newSize )
        {
            return true;
        }

        // Resizing to 0 is a special scenario.  The buffer is de-allocated and set to nullptr.
        if ( i_newSize == 0 )
        {
            TOY_ASSERT( m_buffer != nullptr );
            TOY_VERIFY( MemoryDeallocate< ResidencyT >::Execute( m_buffer ) );
            m_buffer = nullptr;
            m_size   = 0;
            return true;
        }

        // Try an allocate a new buffer.
        void* newBuffer = MemoryAllocate< ResidencyT >::Execute( i_newSize * sizeof( ValueT ) );
        if ( newBuffer == nullptr )
        {
            return false;
        }

        // If there is an existing buffer, perform data migration.
        if ( m_buffer != nullptr )
        {
            // Migrate existing buffer.
            size_t elementsToCopy = std::min( m_size, i_newSize );
            bool   result         = MemoryCopy< ResidencyT, ResidencyT >::Execute(
                /* numBytes */ elementsToCopy * sizeof( ValueT ),
                /* src */ m_buffer,
                /* dst */ newBuffer );
            TOY_VERIFY( result );
            TOY_VERIFY( MemoryDeallocate< ResidencyT >::Execute( m_buffer ) );
        }

        // Assign new buffer ptr & size.
        m_buffer = static_cast< ValueT* >( newBuffer );
        m_size   = i_newSize;

        return true;
    }

    // Fill a range within this array with a value.
    inline bool _Fill( size_t i_begin, size_t i_end, const ValueT& i_value )
    {
        TOY_ASSERT( i_begin < i_end );
        TOY_ASSERT( i_end < m_size );
        size_t numElements = i_end - i_begin;
        MemoryFill< ResidencyT >::Execute( numElements, i_value, m_buffer + i_begin );
        return true;
    }

    size_t  m_size   = 0;
    ValueT* m_buffer = nullptr;
};

/// Operator overload for << to enable writing the string representation of \p i_array into an output
/// stream \p o_outputStream.
///
/// \param o_outputStream the output stream to write into.
/// \param i_array the source vector value type.
///
/// \return the output stream.
template < typename ValueT, Residency ResidencyT >
inline std::ostream& operator<<( std::ostream& o_outputStream, const Array< ValueT, ResidencyT >& i_array )
{
    o_outputStream << i_array.GetStr();
    return o_outputStream;
}

TOY_NS_CLOSE
