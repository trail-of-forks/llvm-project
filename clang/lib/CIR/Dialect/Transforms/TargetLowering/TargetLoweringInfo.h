//===---- TargetLoweringInfo.h - Per-target transform-pass info -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sibling of clang::CIRGen::TargetCIRGenInfo for the transform-pass time.
// Owns an `ABIInfo` instance and provides target-specific hooks the
// CIR-to-CIR rewriter needs (e.g., address-space mapping).
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETLOWERINGINFO_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETLOWERINGINFO_H

#include "ABIInfo.h"
#include <memory>

namespace cir {

class TargetLoweringInfo {
  std::unique_ptr<ABIInfo> info;

public:
  TargetLoweringInfo(std::unique_ptr<ABIInfo> info);
  virtual ~TargetLoweringInfo();

  const ABIInfo &getABIInfo() const { return *info; }
};

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_TARGETLOWERINGINFO_H
