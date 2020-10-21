#pragma once

/// \file base/typeName.h
///
/// Utilities for extracting the names of types.

#include <toy/base/diagnostic.h>
#include <toy/toy.h>

#include <cxxabi.h>
#include <string>

TOY_NS_OPEN

template < typename T >
std::string DemangledTypeName()
{
    int   status;
    char* demangled = abi::__cxa_demangle( typeid( T ).name(), 0, 0, &status );
    TOY_VERIFY( status );
    std::string typeName = demangled;
    free( demangled );
    return typeName;
}

TOY_NS_CLOSE
