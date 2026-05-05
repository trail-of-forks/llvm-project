//===---- LowerTypes.h - Transform-pass type/context helper -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loose equivalent of clang::CIRGen::CIRGenTypes for the transform pass.
// Provides target/data-layout/MLIR-context access to ABI classifiers
// without needing to thread LowerModule through every call.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERTYPES_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERTYPES_H

#include "mlir/IR/MLIRContext.h"

namespace clang {
class TargetInfo;
}

namespace cir {

class LowerModule;

class LowerTypes {
  LowerModule &lm;

public:
  LowerTypes(LowerModule &lm) : lm(lm) {}

  LowerModule &getLowerModule() const { return lm; }
  mlir::MLIRContext *getMLIRContext() const;
  const clang::TargetInfo &getTarget() const;
};

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERTYPES_H
