//===- MissingReturn.h - CIR Missing Return Analysis --------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Detects non-void functions that don't return on all control paths by checking
// cir.return is_implicit attribute (no CFG needed).
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_ANALYSIS_MISSINGRETURN_H
#define CLANG_CIR_ANALYSIS_MISSINGRETURN_H

#include "mlir/IR/BuiltinOps.h"

namespace clang {
class DiagnosticsEngine;
class SourceManager;
} // namespace clang

namespace cir {

/// Diagnose non-void functions that fail to return a value on all control
/// paths.
///
/// Walks all cir::FuncOp in the module, inspects cir::ReturnOp ops for the
/// is_implicit UnitAttr (set by CIRGen when the compiler inserts an implicit
/// return), and emits diag::warn_falloff_nonvoid when a non-void function
/// has an implicit return.
///
/// \param Module  The CIR module to analyze.
/// \param Diags   The diagnostics engine for emitting warnings.
/// \param SM      The source manager for location conversion.
void diagnoseMissingReturn(mlir::ModuleOp Module,
                           clang::DiagnosticsEngine &Diags,
                           clang::SourceManager &SM);

} // namespace cir

#endif // CLANG_CIR_ANALYSIS_MISSINGRETURN_H
