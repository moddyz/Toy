#pragma once

/// \file utils/bitMask.h
///
/// Macro definitions for simplifying the process the operator overloads for a enum
/// type intended to function as a bit mask.

#define TOY_ENUM_BITMASK_OPERATORS( ENUM_TYPE )                                                                        \
    inline ENUM_TYPE& operator|=( ENUM_TYPE& o_enumValue, ENUM_TYPE b )                                                \
    {                                                                                                                  \
        return o_enumValue = static_cast< ENUM_TYPE >( o_enumValue | b );                                              \
    }                                                                                                                  \
                                                                                                                       \
    inline ENUM_TYPE& operator&=( ENUM_TYPE& o_enumValue, ENUM_TYPE b )                                                \
    {                                                                                                                  \
        return o_enumValue = static_cast< ENUM_TYPE >( o_enumValue & b );                                              \
    }                                                                                                                  \
                                                                                                                       \
    inline ENUM_TYPE& operator~( ENUM_TYPE& o_enumValue )                                                              \
    {                                                                                                                  \
        o_enumValue = static_cast< ENUM_TYPE >( ~static_cast< char >( o_enumValue ) );                                 \
        return o_enumValue;                                                                                            \
    }
