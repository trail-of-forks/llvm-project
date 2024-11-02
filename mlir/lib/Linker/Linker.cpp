//===- lib/Linker/Linker.cpp - Module Linker Implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MLIR module linker.
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/Linker.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
