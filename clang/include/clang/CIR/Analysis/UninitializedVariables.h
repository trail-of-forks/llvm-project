//===- UninitializedVariables.h - CIR Uninit Variable Analysis -----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Detects uses of uninitialized local variables via
// DenseForwardDataFlowAnalysis on flattened CIR. Tracks cir.alloca / cir.store
// / cir.load to determine whether variables are initialized before use.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_ANALYSIS_UNINITIALIZEDVARIABLES_H
#define CLANG_CIR_ANALYSIS_UNINITIALIZEDVARIABLES_H

#include "mlir/IR/BuiltinOps.h"

namespace clang {
class DiagnosticsEngine;
class SourceManager;
} // namespace clang

namespace cir {

/// Diagnose uses of uninitialized local variables.
///
/// Clones the module and flattens CIR (cir.break/continue lack
/// RegionBranchTerminatorOpInterface, which crashes DataFlowSolver on
/// structured CIR with loops), then runs MLIR DataFlowSolver with a custom
/// DenseForwardDataFlowAnalysis per function. Emits diag::warn_uninit_var
/// for definitely uninitialized uses and diag::warn_maybe_uninit_var for
/// possibly uninitialized uses.
///
/// \param Module  The CIR module to analyze (not modified; a clone is used).
/// \param Diags   The diagnostics engine for emitting warnings.
/// \param SM      The source manager for location conversion.
void diagnoseUninitializedVariables(mlir::ModuleOp Module,
                                    clang::DiagnosticsEngine &Diags,
                                    clang::SourceManager &SM);

} // namespace cir

#endif // CLANG_CIR_ANALYSIS_UNINITIALIZEDVARIABLES_H
