//===- LowerTypes.cpp - Transform-pass type/context helper ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LowerTypes.h"
#include "LowerModule.h"

namespace cir {

mlir::MLIRContext *LowerTypes::getMLIRContext() const {
  return lm.getMLIRContext();
}

const clang::TargetInfo &LowerTypes::getTarget() const {
  return lm.getTarget();
}

} // namespace cir
