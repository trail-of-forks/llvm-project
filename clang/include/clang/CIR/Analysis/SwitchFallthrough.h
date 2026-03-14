//===- SwitchFallthrough.h - CIR Switch Fallthrough Analysis ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Detects unannotated fall-through between case labels in cir.switch ops
// by walking the CIR region structure directly (no CFG needed).
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_ANALYSIS_SWITCHFALLTHROUGH_H
#define CLANG_CIR_ANALYSIS_SWITCHFALLTHROUGH_H

#include "mlir/IR/BuiltinOps.h"

namespace clang {
class DiagnosticsEngine;
class SourceManager;
} // namespace clang

namespace cir {

/// Diagnose unannotated fall-through between switch labels in CIR.
///
/// Walks all cir::FuncOp in the module, finds cir::SwitchOp ops in simple
/// form, and emits diag::warn_unannotated_fallthrough (or the per-function
/// variant) when a case region terminates with cir.yield (fall-through)
/// and has a non-empty body.
///
/// \param Module     The CIR module to analyze.
/// \param Diags      The diagnostics engine for emitting warnings.
/// \param SM         The source manager for location conversion.
/// \param PerFunction If true, use warn_unannotated_fallthrough_per_function.
void diagnoseSwitchFallthrough(mlir::ModuleOp Module,
                               clang::DiagnosticsEngine &Diags,
                               clang::SourceManager &SM, bool PerFunction);

} // namespace cir

#endif // CLANG_CIR_ANALYSIS_SWITCHFALLTHROUGH_H
