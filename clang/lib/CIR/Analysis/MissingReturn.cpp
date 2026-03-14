//===- MissingReturn.cpp - CIR Missing Return Analysis ------------*- C++
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
// Algorithm:
//   1. Walk all cir::FuncOp in the module.
//   2. Skip declarations and void functions.
//   3. For each function body, walk all cir::ReturnOp ops.
//   4. If any ReturnOp has is_implicit set, the function has a path without
//      an explicit return. Emit warn_falloff_nonvoid.
//   5. If explicit returns also exist, emit the "in all control paths" variant.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Analysis/MissingReturn.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/CIR/Analysis/CIRAnalysisUtils.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace clang;

void cir::diagnoseMissingReturn(ModuleOp Module,
                                clang::DiagnosticsEngine &Diags,
                                clang::SourceManager &SM) {
  Module.walk([&](cir::FuncOp FuncOp) {
    // Skip declarations (no body to analyze).
    if (FuncOp.isDeclaration())
      return;

    // Skip void functions (no return value expected).
    if (FuncOp.getFunctionType().getReturnTypes().empty())
      return;

    // Walk the function body looking for implicit and explicit returns.
    bool HasImplicitReturn = false;
    bool HasExplicitReturn = false;
    cir::ReturnOp ImplicitReturnOp = nullptr;

    FuncOp.walk([&](cir::ReturnOp RetOp) {
      if (RetOp.getIsImplicit()) {
        HasImplicitReturn = true;
        ImplicitReturnOp = RetOp;
      } else {
        HasExplicitReturn = true;
      }
    });

    // If no implicit return, the function returns on all paths.
    if (!HasImplicitReturn)
      return;

    // Convert MLIR location to Clang SourceLocation.
    clang::SourceLocation Loc =
        mlirLocToClangLoc(ImplicitReturnOp.getLoc(), SM);
    if (Loc.isInvalid())
      return;

    // Emit diagnostic:
    //   Arg 0: FalloffFunctionKind::Function (0)
    //   Arg 1: 0 = AlwaysFallThrough, 1 = MaybeFallThrough
    Diags.Report(Loc, diag::warn_falloff_nonvoid)
        << 0 << static_cast<unsigned>(HasExplicitReturn);
  });
}
