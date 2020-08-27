#pragma once

/// \file utils/falseType.h
///
/// For allowing static assertions with meaningful error messages to be
/// raised for certain template specializations.
///
/// static_assert( false, ... ) does not work - because it does not depend on any of
/// the template paramteers, thus is evaluated by the compiler even if the template
/// specialization is not being called anywhere!

#include <toy/toy.h>

#include <type_traits>

TOY_NS_OPEN

template < typename T >
struct FalseType : std::false_type
{
};

TOY_NS_CLOSE
