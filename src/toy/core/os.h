#pragma once

/// \file base/os.h
///
/// Operating system variant definitions.

#if defined( _WIN32 ) || defined( _WIN64 )
#define TOY_WINDOWS
#elif defined( __linux__ )
#define TOY_LINUX
#elif defined( __APPLE__ ) && defined( __MACH__ )
#define TOY_OSX
#endif
