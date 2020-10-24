#pragma once

/// \file base/typeName.h
///
/// Utilities for extracting the names of types.

#include <tri/base/diagnostic.h>
#include <tri/tri.h>

#include <cxxabi.h>
#include <string>

TRI_NS_OPEN

template < typename T >
std::string DemangledTypeName()
{
    int   status;
    char* demangled = abi::__cxa_demangle( typeid( T ).name(), 0, 0, &status );
    TRI_VERIFY( status );
    std::string typeName = demangled;
    free( demangled );
    return typeName;
}

TRI_NS_CLOSE
