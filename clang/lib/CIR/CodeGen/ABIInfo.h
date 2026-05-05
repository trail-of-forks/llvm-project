//===----- ABIInfo.h - ABI information access & encapsulation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_ABIINFO_H
#define LLVM_CLANG_LIB_CIR_ABIINFO_H

#include "clang/AST/CanonicalType.h"
#include "clang/CIR/ABIArgInfo.h"

namespace clang::CIRGen {

class CIRGenFunctionInfo;
class CIRGenTypes;

/// Target-independent base class for per-target ABI classification.
///
/// Subclasses (X8664ABIInfo, ARMABIInfo, ...) override the classify*
/// methods to implement per-target rules. The default base implementation
/// is "everything direct" so that targets without a real classifier yet
/// still produce correct CIR for scalar pass-through cases.
class ABIInfo {
  ABIInfo() = delete;

public:
  CIRGenTypes &cgt;

  ABIInfo(CIRGenTypes &cgt) : cgt(cgt) {}

  virtual ~ABIInfo();

  /// Compute and store ABIArgInfo for the return value and each argument
  /// in `fi`. Default implementation marks everything `Direct` — targets
  /// override to apply real rules (sext/zext for small integers, indirect
  /// for large aggregates, etc.).
  virtual void computeInfo(CIRGenFunctionInfo &fi) const;

  /// Classify a return type for ABI purposes. Default: Direct.
  virtual cir::ABIArgInfo classifyReturnType(clang::CanQualType retTy) const;

  /// Classify an argument type for ABI purposes. Default: Direct.
  virtual cir::ABIArgInfo classifyArgumentType(clang::CanQualType argTy) const;
};

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_ABIINFO_H
