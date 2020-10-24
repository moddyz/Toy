#pragma once

/// \file tri/tri.h
///
/// Building blocks for generating images.

/// \def TRI_NS
///
/// The namespace hosting all the symbols in the tri library.
#define TRI_NS tri

/// \def TRI_NS_USING
///
/// Convenience using directive for TRI_NS.
#define TRI_NS_USING using namespace TRI_NS;

/// \def TRI_NS_OPEN
///
/// Used throughout the library to open the tri namespace scope.
#define TRI_NS_OPEN                                                                                                    \
    namespace TRI_NS                                                                                                   \
    {
/// \def TRI_NS_CLOSE
///
/// Used throughout the library to close the tri namespace scope.
#define TRI_NS_CLOSE }
