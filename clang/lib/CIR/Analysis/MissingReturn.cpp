//===- MissingReturn.cpp - CIR Missing Return Analysis ------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Detects non-void functions that don't return on all control paths by
// checking cir.return is_implicit attribute combined with structured CIR
// reachability analysis.
//
// Algorithm:
//   1. Walk all cir::FuncOp in the module.
//   2. Skip declarations and void functions.
//   3. Find any cir::ReturnOp with is_implicit set.
//   4. Check if the implicit return is reachable via structured control flow
//      walk (noreturn calls, infinite loops, all-paths-return analysis).
//   5. Only emit warn_falloff_nonvoid if the implicit return is reachable.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Analysis/MissingReturn.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/CIR/Analysis/CIRAnalysisUtils.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace clang;

//===----------------------------------------------------------------------===//
// Structured CIR reachability analysis
//
// Determines whether control can "fall through" a structured CIR op to the
// next op in its block. This leverages CIR's nested region structure to do
// reachability via tree traversal rather than CFG graph algorithms.
//===----------------------------------------------------------------------===//

static bool canRegionFallThrough(Region &R);

/// Returns true if the given region contains a cir::BreakOp at the current
/// loop/switch nesting level (does not recurse into nested loops or switches,
/// since their breaks target those constructs, not the enclosing one).
static bool containsBreak(Region &R) {
  for (Block &B : R) {
    for (Operation &Op : B) {
      if (isa<cir::BreakOp>(Op))
        return true;
      // Breaks inside nested loops/switches target those constructs.
      if (isa<cir::WhileOp, cir::ForOp, cir::DoWhileOp, cir::SwitchOp>(Op))
        continue;
      for (Region &Sub : Op.getRegions())
        if (containsBreak(Sub))
          return true;
    }
  }
  return false;
}

/// Returns true if a loop's condition region always evaluates to true
/// (constant true pattern: cir.const #true -> cir.condition).
static bool isConstantTrueCondition(Region &CondRegion) {
  if (CondRegion.empty())
    return false;
  Block &B = CondRegion.front();
  auto CondOp = dyn_cast<cir::ConditionOp>(B.getTerminator());
  if (!CondOp)
    return false;

  Value Cond = CondOp.getCondition();
  // Look through int_to_bool cast.
  if (auto Cast = Cond.getDefiningOp<cir::CastOp>())
    Cond = Cast.getSrc();

  auto ConstOp = Cond.getDefiningOp<cir::ConstantOp>();
  if (!ConstOp)
    return false;

  if (auto BoolAttr = dyn_cast<cir::BoolAttr>(ConstOp.getValue()))
    return BoolAttr.getValue();
  if (auto IntAttr = dyn_cast<cir::IntAttr>(ConstOp.getValue()))
    return !IntAttr.getValue().isZero();
  return false;
}

/// Returns true if control can flow past the given operation to the next
/// operation in the block. This is the core of the structured reachability
/// analysis.
static bool canFallPast(Operation *Op) {
  // Noreturn calls never return.
  if (isa<cir::CallOp>(Op) && Op->hasAttr("noreturn"))
    return false;

  // Trap and unreachable are terminal.
  if (isa<cir::TrapOp, cir::UnreachableOp>(Op))
    return false;

  // Scope: falls through iff its body falls through.
  if (auto Scope = dyn_cast<cir::ScopeOp>(Op))
    return Scope.getScopeRegion().empty() ||
           canRegionFallThrough(Scope.getScopeRegion());

  // If: falls through if either branch can fall through (or no else).
  if (auto If = dyn_cast<cir::IfOp>(Op)) {
    if (If.getElseRegion().empty())
      return true;
    return canRegionFallThrough(If.getThenRegion()) ||
           canRegionFallThrough(If.getElseRegion());
  }

  // While: infinite loop with no break = can't fall through.
  if (auto While = dyn_cast<cir::WhileOp>(Op)) {
    if (isConstantTrueCondition(While.getCond()) &&
        !containsBreak(While.getBody()))
      return false;
    return true;
  }

  // For: same as while.
  if (auto For = dyn_cast<cir::ForOp>(Op)) {
    if (isConstantTrueCondition(For.getCond()) && !containsBreak(For.getBody()))
      return false;
    return true;
  }

  // DoWhile: if body always terminates (e.g. return), loop never reaches
  // condition. Also handle infinite do-while with no break.
  if (auto DoWhile = dyn_cast<cir::DoWhileOp>(Op)) {
    if (!DoWhile.getBody().empty() && !canRegionFallThrough(DoWhile.getBody()))
      return false;
    if (isConstantTrueCondition(DoWhile.getCond()) &&
        !containsBreak(DoWhile.getBody()))
      return false;
    return true;
  }

  // Switch: can't fall through if all cases are covered and none fall through.
  if (auto Switch = dyn_cast<cir::SwitchOp>(Op)) {
    llvm::SmallVector<cir::CaseOp> Cases;
    Switch.collectCases(Cases);

    bool HasDefault = false;
    bool AllEnumCovered = Switch.getAllEnumCasesCovered();
    for (auto &CaseOp : Cases) {
      if (CaseOp.getKind() == cir::CaseOpKind::Default)
        HasDefault = true;
      if (canRegionFallThrough(CaseOp.getCaseRegion()))
        return true;
    }
    // No case falls through. If all values are covered, switch can't fall
    // through either.
    if (HasDefault || AllEnumCovered)
      return false;
    return true;
  }

  return true;
}

/// Returns true if control can fall through the given region, i.e. reach a
/// cir.yield that returns control to the parent op. Handles multi-block
/// regions by checking all blocks' terminators.
static bool canRegionFallThrough(Region &R) {
  if (R.empty())
    return true;
  for (Block &B : R) {
    // Check non-terminator ops for guaranteed non-fallthrough.
    for (Operation &Op : B.without_terminator())
      if (!canFallPast(&Op))
        return false;
    // Check the terminator.
    Operation *Term = B.getTerminator();
    if (!Term)
      continue;
    // A yield means control returns to the parent — region falls through.
    if (isa<cir::YieldOp>(Term))
      return true;
  }
  // No yield found — all paths exit via return/break/continue/branch-to-return.
  return false;
}

/// Returns true if the implicit return at the end of a function is reachable
/// via structured control flow analysis.
static bool isImplicitReturnReachable(cir::FuncOp Func) {
  Block &Entry = Func.getBody().front();
  for (Operation &Op : Entry) {
    // Stop before the implicit return itself (it's the terminator).
    if (isa<cir::ReturnOp>(Op))
      break;
    if (!canFallPast(&Op))
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Main analysis entry point
//===----------------------------------------------------------------------===//

void cir::diagnoseMissingReturn(ModuleOp Module,
                                clang::DiagnosticsEngine &Diags,
                                clang::SourceManager &SM) {
  Module.walk([&](cir::FuncOp FuncOp) {
    if (FuncOp.isDeclaration())
      return;

    if (FuncOp.getFunctionType().getReturnTypes().empty())
      return;

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

    if (!HasImplicitReturn)
      return;

    // Check if the implicit return is actually reachable.
    if (!isImplicitReturnReachable(FuncOp))
      return;

    clang::SourceLocation Loc =
        mlirLocToClangLoc(ImplicitReturnOp.getLoc(), SM);
    if (Loc.isInvalid())
      return;

    // Arg 0: FalloffFunctionKind::Function (0)
    // Arg 1: 0 = AlwaysFallThrough, 1 = MaybeFallThrough
    Diags.Report(Loc, diag::warn_falloff_nonvoid)
        << 0 << static_cast<unsigned>(HasExplicitReturn);
  });
}
