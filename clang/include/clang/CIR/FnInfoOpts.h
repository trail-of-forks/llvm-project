//===---- FnInfoOpts.h - Options for arrange*FunctionInfo -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_FNINFOOPTS_H
#define LLVM_CLANG_CIR_FNINFOOPTS_H

#include "llvm/ADT/BitmaskEnum.h"

namespace cir {

/// Per-call-site options consulted by both the CIRGen-time and
/// transform-pass-time arrange*FunctionInfo paths. Bitmask enum so callers
/// can OR flags together as needed.
enum class FnInfoOpts {
  None = 0,
  IsInstanceMethod = 1 << 0,
  IsChainCall = 1 << 1,
  IsDelegateCall = 1 << 2,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/IsDelegateCall),
};

} // namespace cir

#endif // LLVM_CLANG_CIR_FNINFOOPTS_H
