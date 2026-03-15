//===- LifetimeSafety.h - CIR Lifetime Safety Analysis -----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Detects return-of-local-address patterns via DenseForwardDataFlowAnalysis
// on flattened CIR. Tracks pointer-to-alloca alias relationships through
// cir.alloca / cir.store / cir.load / cir.cast to determine whether a
// returned pointer references stack memory.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_ANALYSIS_LIFETIMESAFETY_H
#define CLANG_CIR_ANALYSIS_LIFETIMESAFETY_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
class DiagnosticsEngine;
class SourceManager;
} // namespace clang

namespace cir {

/// Per-function summary of lifetime-relevant aliasing, consumed by
/// cross-function analysis in Phase 15.
struct FuncLifetimeSummary {
  /// Parameter indices whose alloca address flows to the return value.
  llvm::SmallVector<unsigned> ParamAliasesReturn;
  /// True if the function returns the address of a local variable.
  bool ReturnsLocalAddress = false;
};

/// Diagnose lifetime safety violations (return-of-local-address).
///
/// Clones the module and flattens CIR (cir.break/continue lack
/// RegionBranchTerminatorOpInterface, which crashes DataFlowSolver on
/// structured CIR with loops), then runs MLIR DataFlowSolver with a custom
/// DenseForwardDataFlowAnalysis per function. Emits
/// diag::warn_ret_stack_addr_ref for pointers to stack memory that are
/// returned from functions.
///
/// \param Module  The CIR module to analyze (not modified; a clone is used).
/// \param Diags   The diagnostics engine for emitting warnings.
/// \param SM      The source manager for location conversion.
void diagnoseLifetimeSafety(mlir::ModuleOp Module,
                            clang::DiagnosticsEngine &Diags,
                            clang::SourceManager &SM);

} // namespace cir

#endif // CLANG_CIR_ANALYSIS_LIFETIMESAFETY_H
