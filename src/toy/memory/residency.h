#pragma once

/// \file memory/residency.h
///
/// Memory residency.

#include <toy/toy.h>

TOY_NS_OPEN

/// \enum Residency
///
/// The device where memory resides.
enum Residency
{
    Host,
    Cuda
};

TOY_NS_CLOSE
