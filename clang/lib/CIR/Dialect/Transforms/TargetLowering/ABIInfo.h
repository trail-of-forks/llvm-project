//===----- ABIInfo.h - Transform-pass-time ABI classification ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sibling of clang::CIRGen::ABIInfo. The CIRGen-time classifier runs during
// AST-to-CIR lowering and operates on clang::CanQualType. This one runs as
// a CIR-to-CIR transform pass and operates on already-built mlir::Type, so
// it can rewrite signatures (sret/byval/coerce/extend) without touching the
// AST.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFO_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFO_H

namespace cir {

class LowerTypes;
class LowerFunctionInfo;

/// Target-independent base for transform-pass-time ABI classification.
///
/// Targets override `computeInfo` to populate per-arg ABIArgInfos on a
/// LowerFunctionInfo with their own rules (HFAs, byval thresholds, etc.).
class ABIInfo {
protected:
  LowerTypes &lt;

public:
  ABIInfo(LowerTypes &lt) : lt(lt) {}
  virtual ~ABIInfo();

  /// Populate ABIArgInfos on `fi` per this target's rules.
  virtual void computeInfo(LowerFunctionInfo &fi) const = 0;
};

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFO_H
