#pragma once

/// \file mouse.h
///
/// Mouse related utilities.

#include "bitMask.h"

/// \enum MouseButton
///
/// Mouse button bit flags.
///
/// This is used by the \ref Window class for providing mouse button pressed
/// state.
enum MouseButton : char
{
    MouseButton_None = 0,        // binary 0000
    MouseButton_Left = 1 << 0,   // binary 0001
    MouseButton_Middle = 1 << 1, // binary 0010
    MouseButton_Right = 1 << 2,  // binary 0100
};

TRI_ENUM_BITMASK_OPERATORS(MouseButton);
