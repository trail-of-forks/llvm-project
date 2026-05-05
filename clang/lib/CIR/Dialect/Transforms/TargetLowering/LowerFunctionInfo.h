//===- LowerFunctionInfo.h - Transform-pass per-call ABI metadata -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sibling of clang::CIRGen::CIRGenFunctionInfo for the transform-pass time.
// Carries the same shape (return type + N argument types, plus per-slot
// ABIArgInfo) but stores mlir::Type instead of clang::CanQualType so the
// transform pass can operate on already-built CIR.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTIONINFO_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTIONINFO_H

#include "mlir/IR/Types.h"
#include "clang/CIR/ABIArgInfo.h"
#include "llvm/ADT/SmallVector.h"

namespace cir {

class LowerFunctionInfo {
public:
  struct ArgInfo {
    mlir::Type type;
    cir::ABIArgInfo info;
  };

private:
  mlir::Type returnType;
  cir::ABIArgInfo returnInfo;
  llvm::SmallVector<ArgInfo, 8> args;
  bool variadic = false;

public:
  LowerFunctionInfo() = default;

  void setReturnType(mlir::Type ty) { returnType = ty; }
  mlir::Type getReturnType() const { return returnType; }
  cir::ABIArgInfo &getReturnInfo() { return returnInfo; }
  const cir::ABIArgInfo &getReturnInfo() const { return returnInfo; }

  void addArgument(mlir::Type ty) { args.push_back({ty, {}}); }
  llvm::MutableArrayRef<ArgInfo> arguments() { return args; }
  llvm::ArrayRef<ArgInfo> arguments() const { return args; }

  unsigned getNumArgs() const { return args.size(); }
  void setVariadic(bool v) { variadic = v; }
  bool isVariadic() const { return variadic; }
};

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTIONINFO_H
