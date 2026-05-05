//===---- ARM.h - ARM target abstractions for ClangIR -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_TARGET_ARM_H
#define LLVM_CLANG_CIR_TARGET_ARM_H

namespace cir {

/// The ABI kind for ARM targets.
enum class ARMABIKind {
  APCS = 0,
  AAPCS,
  AAPCS_VFP,
  AAPCS16_VFP,
};

} // namespace cir

#endif // LLVM_CLANG_CIR_TARGET_ARM_H
