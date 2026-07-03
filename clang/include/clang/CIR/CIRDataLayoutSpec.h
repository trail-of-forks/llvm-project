//===--- CIRDataLayoutSpec.h - CIR data layout spec emission ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helper for deriving a CIR module's `dlti.dl_spec` attribute from an
// LLVM data layout, used by CIRGen and cir-translate.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_CIRDATALAYOUTSPEC_H
#define CLANG_CIR_CIRDATALAYOUTSPEC_H

#include "mlir/IR/BuiltinOps.h"

namespace llvm {
class DataLayout;
} // namespace llvm

namespace cir {

/// Translate `dl` into a DLTI data layout spec, append a
/// `!cir.ptr<!cir.void>`-keyed `#cir.ptr_spec` entry mirroring the
/// `!llvm.ptr` (address space 0) pointer entry, and set the result as the
/// module's `dlti.dl_spec` attribute. Only the default address space is
/// supported for now.
void setCIRDataLayout(mlir::ModuleOp mod, const llvm::DataLayout &dl);

} // namespace cir

#endif // CLANG_CIR_CIRDATALAYOUTSPEC_H
