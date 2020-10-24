#pragma once

/// \file base/log.h
///
/// Logging utility functions.

#include <stdio.h>

/// \def TRI_LOG_DEBUG( msgFormat, ... )
///
/// Logs a formatted message at the \em DEBUG level.
#define TRI_LOG_DEBUG( msgFormat, ... ) printf( msgFormat, ##__VA_ARGS__ );

/// \def TRI_LOG_INFO( msgFormat, ... )
///
/// Logs a formatted message at the \em INFO level.
#define TRI_LOG_INFO( msgFormat, ... ) printf( msgFormat, ##__VA_ARGS__ );

/// \def TRI_LOG_WARN( msgFormat, ... )
///
/// Logs a formatted message at the \em WARN level.
#define TRI_LOG_WARN( msgFormat, ... ) printf( msgFormat, ##__VA_ARGS__ );

/// \def TRI_LOG_ERROR( msgFormat, ... )
///
/// Logs a formatted message at the \em ERROR level.
#define TRI_LOG_ERROR( msgFormat, ... ) fprintf( stderr, msgFormat, ##__VA_ARGS__ );
