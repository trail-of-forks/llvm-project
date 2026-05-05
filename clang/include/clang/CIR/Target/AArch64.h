//===---- AArch64.h - AArch64 target abstractions for ClangIR ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_TARGET_AARCH64_H
#define LLVM_CLANG_CIR_TARGET_AARCH64_H

namespace cir {

/// The ABI kind for AArch64 targets.
enum class AArch64ABIKind {
  AAPCS = 0,
  DarwinPCS,
  Win64,
  AAPCSSoft,
};

} // namespace cir

#endif // LLVM_CLANG_CIR_TARGET_AARCH64_H
