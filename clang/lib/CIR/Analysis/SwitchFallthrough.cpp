//===- SwitchFallthrough.cpp - CIR Switch Fallthrough Analysis ----*- C++
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
// Algorithm:
//   1. Walk all cir::FuncOp in the module.
//   2. For each function, walk to find all cir::SwitchOp ops.
//   3. For each SwitchOp in simple form, examine consecutive case regions.
//   4. If a case region's last terminator is cir.yield (fall-through) and the
//      case body is non-empty, emit warn_unannotated_fallthrough at the NEXT
//      case label's location.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Analysis/SwitchFallthrough.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/CIR/Analysis/CIRAnalysisUtils.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace clang;

/// Check if a case region has a non-empty body that falls through.
///
/// A case region falls through when its last block terminates with cir.yield
/// AND the region contains at least one operation other than the terminating
/// yield. Empty cases (consecutive labels like `case 1: case 2:`) produce
/// regions containing only cir.yield and should not trigger a warning.
static bool caseHasFallthrough(cir::CaseOp CaseOp) {
  Region &CaseRegion = CaseOp.getCaseRegion();
  if (CaseRegion.empty())
    return false;

  // Get the last block in the case region.
  Block &LastBlock = CaseRegion.back();
  if (LastBlock.empty())
    return false;

  Operation *Terminator = LastBlock.getTerminator();
  if (!Terminator)
    return false;

  // Only cir.yield in a case region signals fall-through.
  if (!isa<cir::YieldOp>(Terminator))
    return false;

  // Check if this is an empty case body (only a cir.yield, no other ops
  // across all blocks in the region). Empty cases like `case 1: case 2:`
  // should not trigger a warning.
  for (Block &B : CaseRegion) {
    for (Operation &Op : B) {
      if (&Op != Terminator)
        return true; // Non-trivial content found -- this is a real fallthrough.
    }
  }
  return false; // Empty case body.
}

void cir::diagnoseSwitchFallthrough(ModuleOp Module,
                                    clang::DiagnosticsEngine &Diags,
                                    clang::SourceManager &SM,
                                    bool PerFunction) {
  unsigned DiagID = PerFunction
                        ? clang::diag::warn_unannotated_fallthrough_per_function
                        : clang::diag::warn_unannotated_fallthrough;

  Module.walk([&](cir::FuncOp FuncOp) {
    FuncOp.walk([&](cir::SwitchOp SwitchOp) {
      llvm::SmallVector<cir::CaseOp> Cases;
      if (!SwitchOp.isSimpleForm(Cases))
        return; // Skip non-simple switches (e.g. Duff's device).

      for (size_t I = 0; I + 1 < Cases.size(); ++I) {
        if (caseHasFallthrough(Cases[I])) {
          // Emit warning at the NEXT case label (the one being fallen into).
          clang::SourceLocation Loc =
              cir::mlirLocToClangLoc(Cases[I + 1].getLoc(), SM);
          if (Loc.isInvalid())
            continue;

          Diags.Report(Loc, DiagID);
          Diags.Report(Loc, clang::diag::note_insert_break_fixit)
              << clang::FixItHint::CreateInsertion(Loc, "break; ");
        }
      }
    });
  });
}
