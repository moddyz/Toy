#pragma once

#include <toy/toy.h>

#include <toy/base/log.h>

#include <stdarg.h>

/// \file base/diagnostic.h
///
/// Diagnostic utilities.

/// \def TOY_ASSERT( expr )
///
/// Asserts that the expression \p expr is \em true in debug builds. If \p expr evalutes \em false,
/// an error message will be printed with contextual information including the failure site.
///
/// In release builds, this is compiled out.
#ifdef TOY_DEBUG
#define TOY_ASSERT( expr )                                                                                             \
    if ( !( expr ) )                                                                                                   \
    {                                                                                                                  \
        TOY_NS::_Assert( #expr, __FILE__, __LINE__ );                                                                  \
    }
#else
#define TOY_ASSERT( expr, ... ) void()
#endif

/// \def TOY_ASSERT_MSG( expr, format, ... )
///
/// Similar to \ref TOY_ASSERT, with the addition of a formatted message when the expression \p expr fails to evaluate
/// in debug builds.
#ifdef TOY_DEBUG
#define TOY_ASSERT_MSG( expr, format, ... )                                                                            \
    if ( !( expr ) )                                                                                                   \
    {                                                                                                                  \
        TOY_NS::_Assert( #expr, __FILE__, __LINE__ );                                                                  \
        TOY_LOG_ERROR( format, ##__VA_ARGS__ );                                                                        \
    }
#else
#define TOY_ASSERT_MSG( expr, format, ... ) void()
#endif

/// \def TOY_VERIFY( expr )
///
/// Verifies that the expression \p expr evaluates to \em true at runtime. If \p expr evalutes \em false,
/// an error message will be printed with contextual information including the failure site.
///
/// TOY_VERIFY is different from \ref TOY_ASSERT in that it does \em not get compiled out for release builds,
/// so use sparingly!
#define TOY_VERIFY( expr )                                                                                             \
    if ( !( expr ) )                                                                                                   \
    {                                                                                                                  \
        TOY_NS::_Assert( #expr, __FILE__, __LINE__ );                                                                  \
    }

/// \def TOY_VERIFY_MSG( expr, format, ... )
///
/// Similar to \ref TOY_VERIFY, with the addition of a formatted message when the expression \p expr fails to evaluate.
#define TOY_VERIFY_MSG( expr, format, ... )                                                                            \
    if ( !( expr ) )                                                                                                   \
    {                                                                                                                  \
        TOY_NS::_Verify( #expr, __FILE__, __LINE__ );                                                                  \
        TOY_LOG_ERROR( format, ##__VA_ARGS__ );                                                                        \
    }

TOY_NS_OPEN

/// Not intended to be used directly, \ref TOY_ASSERT instead.
inline void _Assert( const char* i_expression, const char* i_file, size_t i_line )
{
    TOY_LOG_ERROR( "Assertion failed for expression: %s, at %s:%lu\n", i_expression, i_file, i_line );
}

/// Not intended to be used directly, \ref TOY_VERIFY instead.
inline void _Verify( const char* i_expression, const char* i_file, size_t i_line )
{
    TOY_LOG_ERROR( "Verification failed for expression: %s, at %s:%lu\n", i_expression, i_file, i_line );
}

TOY_NS_CLOSE
