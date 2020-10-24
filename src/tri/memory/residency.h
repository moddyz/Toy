#pragma once

/// \file memory/residency.h
///
/// Memory residency.

#include <tri/tri.h>

TRI_NS_OPEN

/// \enum Residency
///
/// The device where memory resides.
enum Residency
{
    Host,
    CUDA
};

TRI_NS_CLOSE
