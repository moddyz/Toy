#pragma once

/// \file base/diagnostic.h
///
/// Diagnostic utilities.

#include <stdarg.h>

#include <tri/base/log.h>
#include <tri/tri.h>

/// \def TRI_ASSERT( expr )
///
/// Asserts that the expression \p expr is \em true in debug builds. If \p expr evalutes \em false,
/// an error message will be printed with contextual information including the failure site.
///
/// In release builds, this is compiled out.
#ifdef TRI_DEBUG
#define TRI_ASSERT( expr )                                                                                             \
    if ( !( expr ) )                                                                                                   \
    {                                                                                                                  \
        TRI_NS::_Assert( #expr, __FILE__, __LINE__ );                                                                  \
    }
#else
#define TRI_ASSERT( expr, ... ) void()
#endif

/// \def TRI_ASSERT_MSG( expr, format, ... )
///
/// Similar to \ref TRI_ASSERT, with the addition of a formatted message when the expression \p expr fails to evaluate
/// in debug builds.
#ifdef TRI_DEBUG
#define TRI_ASSERT_MSG( expr, format, ... )                                                                            \
    if ( !( expr ) )                                                                                                   \
    {                                                                                                                  \
        TRI_NS::_Assert( #expr, __FILE__, __LINE__ );                                                                  \
        TRI_LOG_ERROR( format, ##__VA_ARGS__ );                                                                        \
    }
#else
#define TRI_ASSERT_MSG( expr, format, ... ) void()
#endif

/// \def TRI_VERIFY( expr )
///
/// Verifies that the expression \p expr evaluates to \em true at runtime. If \p expr evalutes \em false,
/// an error message will be printed with contextual information including the failure site.
///
/// TRI_VERIFY is different from \ref TRI_ASSERT in that it does \em not get compiled out for release builds,
/// so use sparingly!
#define TRI_VERIFY( expr )                                                                                             \
    if ( !( expr ) )                                                                                                   \
    {                                                                                                                  \
        TRI_NS::_Assert( #expr, __FILE__, __LINE__ );                                                                  \
    }

/// \def TRI_VERIFY_MSG( expr, format, ... )
///
/// Similar to \ref TRI_VERIFY, with the addition of a formatted message when the expression \p expr fails to evaluate.
#define TRI_VERIFY_MSG( expr, format, ... )                                                                            \
    if ( !( expr ) )                                                                                                   \
    {                                                                                                                  \
        TRI_NS::_Verify( #expr, __FILE__, __LINE__ );                                                                  \
        TRI_LOG_ERROR( format, ##__VA_ARGS__ );                                                                        \
    }

TRI_NS_OPEN

/// Not intended to be used directly, \ref TRI_ASSERT instead.
inline void _Assert( const char* i_expression, const char* i_file, size_t i_line )
{
    TRI_LOG_ERROR( "Assertion failed for expression: %s, at %s:%lu\n", i_expression, i_file, i_line );
}

/// Not intended to be used directly, \ref TRI_VERIFY instead.
inline void _Verify( const char* i_expression, const char* i_file, size_t i_line )
{
    TRI_LOG_ERROR( "Verification failed for expression: %s, at %s:%lu\n", i_expression, i_file, i_line );
}

TRI_NS_CLOSE
