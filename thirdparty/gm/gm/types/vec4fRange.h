//
// This file is auto-generated, please do not modify directly!
//

#pragma once

/// \file vec4fRange.h
/// \ingroup gm_types_range

#include <gm/gm.h>

#include <gm/types/vec4f.h>

#include <gm/base/almost.h>

#include <cmath>
#include <limits>
#include <sstream>

GM_NS_OPEN

/// \class Vec4fRange
/// \ingroup gm_types_range
///
/// Class definition for a bounded range of Vec4f(s).
class Vec4fRange final
{
public:
    /// \typedef ValueType
    ///
    /// Convenience type definition of \ref Vec4fRange's elemental value type.
    using ValueType = Vec4f;

    // --------------------------------------------------------------------- //
    /// \name Construction
    // --------------------------------------------------------------------- //

    /// Default constructor, initializing an \em empty range.
    ///
    /// An empty range has a min value with max numerical limit, conversely
    /// a max value with min numerical limit.
    GM_HOST_DEVICE constexpr inline Vec4fRange() = default;

    /// Explicit constructor for initializing a minimum maximum range.
    ///
    /// \param i_min Minimum.
    /// \param i_max Maximum.
    GM_HOST_DEVICE constexpr explicit inline Vec4fRange( const Vec4f& i_min, const Vec4f& i_max )
        : m_min( i_min )
        , m_max( i_max )
    {
    }

    // --------------------------------------------------------------------- //
    /// \name Access
    // --------------------------------------------------------------------- //

    /// Read access to the minimum bound of this range.
    ///
    /// \return The minimum bound.
    GM_HOST_DEVICE inline const Vec4f& Min() const
    {
        return m_min;
    }

    /// Write access to the minimum bound of this range.
    ///
    /// \return The minimum bound.
    GM_HOST_DEVICE inline Vec4f& Min()
    {
        return m_min;
    }

    /// Read access to the maximum bound of this range.
    ///
    /// \return The maximum bound.
    GM_HOST_DEVICE inline const Vec4f& Max() const
    {
        return m_max;
    }

    /// Write access to the maximum bound of this range.
    ///
    /// \return The maximum bound.
    GM_HOST_DEVICE inline Vec4f& Max()
    {
        return m_max;
    }

    // --------------------------------------------------------------------- //
    /// \name Comparison
    // --------------------------------------------------------------------- //

    /// Equality comparison against \p i_range.
    ///
    /// \param i_range The range to compare against.
    ///
    /// \retval true If this range is equal to \p i_range.
    /// \retval false If this range is not equal to \p i_range.
    GM_HOST_DEVICE inline bool operator==( const Vec4fRange& i_range ) const
    {
        return Min() == i_range.Min() && Max() == i_range.Max();
    }

    /// Non-equality comparison against \p i_range.
    ///
    /// \param i_range The range to compare against.
    ///
    /// \retval true If this range is not equal to \p i_range.
    /// \retval false If this range is equal to \p i_range.
    GM_HOST_DEVICE inline bool operator!=( const Vec4fRange& i_range ) const
    {
        return !( ( *this ) == i_range );
    }

    /// Check if this range is empty.
    ///
    /// A range is empty if any of the components in the minimum is greater
    /// than the maximum.
    ///
    /// \retval true If this range is empty.
    /// \retval false If this range is non-empty.
    GM_HOST_DEVICE inline bool IsEmpty() const
    {
        return Min()[ 0 ] > Max()[ 0 ] || Min()[ 1 ] > Max()[ 1 ] || Min()[ 2 ] > Max()[ 2 ] || Min()[ 3 ] > Max()[ 3 ];
    }

    // --------------------------------------------------------------------- //
    /// \name Debug
    // --------------------------------------------------------------------- //

    /// Check if the min or max contain NaN values.
    ///
    /// \return If this range has NaN values.
    GM_HOST_DEVICE inline bool HasNaNs() const
    {
        return Min().HasNaNs() || Max().HasNaNs();
    }

    /// Get the string representation.  For debugging purposes.
    ///
    /// \param i_classPrefix optional string to prefix class tokens.
    ///
    /// \return descriptive string representing this type instance.
    inline std::string GetString( const std::string& i_classPrefix = std::string() ) const
    {
        std::stringstream ss;
        ss << i_classPrefix << "Vec4fRange( ";
        ss << m_min.GetString( i_classPrefix );
        ss << ", ";
        ss << m_max.GetString( i_classPrefix );
        ss << " )";
        return ss.str();
    }

private:
    // Min bound member.
    Vec4f m_min = Vec4f( std::numeric_limits< float >::max(),
                         std::numeric_limits< float >::max(),
                         std::numeric_limits< float >::max(),
                         std::numeric_limits< float >::max() );

    // Max bound member.
    Vec4f m_max = Vec4f( std::numeric_limits< float >::min(),
                         std::numeric_limits< float >::min(),
                         std::numeric_limits< float >::min(),
                         std::numeric_limits< float >::min() );
};

/// Operator overload for << to enable writing the string representation of \p i_value into an output
/// stream \p o_outputStream.
///
/// \param o_outputStream the output stream to write into.
/// \param i_value the source composite value type.
///
/// \return the output stream.
inline std::ostream& operator<<( std::ostream& o_outputStream, const Vec4fRange& i_value )
{
    o_outputStream << i_value.GetString();
    return o_outputStream;
}

GM_NS_CLOSE