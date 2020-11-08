#pragma once

/// \file base/falseType.h
///
/// For allowing static assertions with meaningful error messages to be
/// raised for certain template specializations.
///
/// static_assert( false, ... ) does not work - because it does not depend on any of
/// the template paramteers, thus is evaluated by the compiler even if the template
/// specialization is not being called anywhere!

#include <tri/tri.h>

#include <type_traits>

TRI_NS_OPEN

template < typename T >
struct FalseType : std::false_type
{
};

TRI_NS_CLOSE
