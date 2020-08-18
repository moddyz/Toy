#pragma once

/// \file base/log.h
///
/// Logging utility functions.

#include <stdio.h>

/// \def TOY_LOG_INFO( msgFormat, ... )
///
/// Logs a formatted message at the \em INFO level.
#define TOY_LOG_INFO( msgFormat, ... ) printf( msgFormat, ##__VA_ARGS__ );

/// \def TOY_LOG_WARN( msgFormat, ... )
///
/// Logs a formatted message at the \em WARN level.
#define TOY_LOG_WARN( msgFormat, ... ) printf( msgFormat, ##__VA_ARGS__ );

/// \def TOY_LOG_ERROR( msgFormat, ... )
///
/// Logs a formatted message at the \em ERROR level.
#define TOY_LOG_ERROR( msgFormat, ... ) fprintf( stderr, msgFormat, ##__VA_ARGS__ );
