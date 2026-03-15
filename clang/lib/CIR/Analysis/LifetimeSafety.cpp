//===- LifetimeSafety.cpp - CIR Lifetime Safety Analysis ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Detects return-of-local-address patterns using MLIR's
// DenseForwardDataFlowAnalysis on flattened CIR.
//
// Algorithm:
//   1. Pre-flattening: walk structured CIR to collect scope-to-alloca mappings.
//   2. Clone the module (original must stay structured for downstream passes).
//   3. Flatten CIR via cir-flatten-cfg + goto-solver.
//   4. For each cir::FuncOp, run DataFlowSolver with baseline analyses and
//      LifetimeAnalysis to track pointer-to-alloca alias state.
//   5. Post-solver walk: check cir::ReturnOp ops for returned pointers that
//      alias local allocas. Emit warn_ret_stack_addr_ref diagnostics.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Analysis/LifetimeSafety.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Pass/PassManager.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/CIR/Analysis/CIRAnalysisUtils.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace mlir::dataflow;
using namespace clang;

//===----------------------------------------------------------------------===//
// PtrState and PointerInfo
//===----------------------------------------------------------------------===//

namespace {

/// Tracks what a pointer value points to.
enum class PtrState : uint8_t {
  Unknown = 0,     ///< No alias info (bottom/default).
  PointsToLocal,   ///< Points to a local alloca.
  PointsToParam,   ///< Points to a parameter alloca.
  PointsToUnknown, ///< Points to something we can't track.
  Dangling,        ///< Points to dead stack memory.
};

/// Per-pointer alias information.
struct PointerInfo {
  PtrState State = PtrState::Unknown;
  /// The alloca this pointer targets (valid when PointsToLocal/PointsToParam).
  Value TargetAlloca;

  bool operator==(const PointerInfo &Other) const {
    return State == Other.State && TargetAlloca == Other.TargetAlloca;
  }
  bool operator!=(const PointerInfo &Other) const { return !(*this == Other); }
};

//===----------------------------------------------------------------------===//
// LifetimeLattice
//===----------------------------------------------------------------------===//

/// Dense lattice tracking pointer-to-alloca alias relationships for each
/// pointer-typed SSA value in a function.
class LifetimeLattice : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LifetimeLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &RHS) override {
    const auto &Other = static_cast<const LifetimeLattice &>(RHS);
    ChangeResult Changed = ChangeResult::NoChange;

    // Bottom is the join identity.
    if (Other.AtBottom)
      return ChangeResult::NoChange;

    if (AtBottom) {
      AtBottom = false;
      PtrStates = Other.PtrStates;
      return Other.PtrStates.empty() ? ChangeResult::NoChange
                                     : ChangeResult::Change;
    }

    for (const auto &[Val, OtherInfo] : Other.PtrStates) {
      auto It = PtrStates.find(Val);
      if (It == PtrStates.end()) {
        PtrStates[Val] = OtherInfo;
        Changed = ChangeResult::Change;
      } else if (It->second != OtherInfo) {
        // Merge: if either is Dangling, result is Dangling (conservative).
        if (It->second.State == PtrState::Dangling ||
            OtherInfo.State == PtrState::Dangling) {
          if (It->second.State != PtrState::Dangling) {
            It->second = {PtrState::Dangling, Value()};
            Changed = ChangeResult::Change;
          }
        } else if (It->second.State == PtrState::PointsToLocal &&
                   OtherInfo.State == PtrState::PointsToLocal &&
                   It->second.TargetAlloca != OtherInfo.TargetAlloca) {
          // Different local targets -> can't track precisely.
          It->second = {PtrState::PointsToUnknown, Value()};
          Changed = ChangeResult::Change;
        } else {
          // Mixed states -> unknown.
          It->second = {PtrState::PointsToUnknown, Value()};
          Changed = ChangeResult::Change;
        }
      }
    }

    return Changed;
  }

  void print(raw_ostream &OS) const override {
    OS << "LifetimeLattice{";
    bool First = true;
    for (const auto &[Val, Info] : PtrStates) {
      if (!First)
        OS << ", ";
      First = false;
      Val.print(OS);
      OS << "=";
      switch (Info.State) {
      case PtrState::Unknown:
        OS << "Unknown";
        break;
      case PtrState::PointsToLocal:
        OS << "Local";
        break;
      case PtrState::PointsToParam:
        OS << "Param";
        break;
      case PtrState::PointsToUnknown:
        OS << "PtsUnk";
        break;
      case PtrState::Dangling:
        OS << "Dangling";
        break;
      }
    }
    OS << "}";
  }

  PointerInfo getPointerInfo(Value Ptr) const {
    auto It = PtrStates.find(Ptr);
    return It != PtrStates.end() ? It->second : PointerInfo();
  }

  ChangeResult setPointerInfo(Value Ptr, PointerInfo Info) {
    AtBottom = false;
    auto [It, Inserted] = PtrStates.try_emplace(Ptr, Info);
    if (!Inserted && It->second == Info)
      return ChangeResult::NoChange;
    It->second = Info;
    return ChangeResult::Change;
  }

  ChangeResult reset() {
    ChangeResult Changed = ChangeResult::NoChange;
    if (!PtrStates.empty()) {
      PtrStates.clear();
      Changed = ChangeResult::Change;
    }
    if (!AtBottom) {
      AtBottom = true;
      Changed = ChangeResult::Change;
    }
    return Changed;
  }

private:
  DenseMap<Value, PointerInfo> PtrStates;
  /// True when this lattice is at bottom (identity for join).
  bool AtBottom = true;
};

//===----------------------------------------------------------------------===//
// LifetimeAnalysis (DenseForwardDataFlowAnalysis)
//===----------------------------------------------------------------------===//

class LifetimeAnalysis : public DenseForwardDataFlowAnalysis<LifetimeLattice> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *Op, const LifetimeLattice &Before,
                               LifetimeLattice *After) override {
    ChangeResult Changed = After->join(Before);

    if (auto Alloca = dyn_cast<cir::AllocaOp>(Op)) {
      // Every alloca result is a pointer to its own stack slot.
      Changed |= After->setPointerInfo(
          Alloca.getResult(), {PtrState::PointsToLocal, Alloca.getResult()});
    } else if (auto Store = dyn_cast<cir::StoreOp>(Op)) {
      // If storing a pointer-typed value, propagate alias info from the
      // stored value to the target address. Always update the target --
      // even Unknown values must overwrite stale PointsToLocal info from
      // the alloca definition, so that loading the target later doesn't
      // incorrectly report the alloca itself as returned local memory.
      Value StoredVal = Store.getValue();
      if (mlir::isa<cir::PointerType>(StoredVal.getType())) {
        PointerInfo Info = Before.getPointerInfo(StoredVal);
        Changed |= After->setPointerInfo(Store.getAddr(), Info);
      }
    } else if (auto Load = dyn_cast<cir::LoadOp>(Op)) {
      // If loading from an address that holds pointer alias info, and the
      // loaded value is pointer-typed, propagate alias info.
      if (mlir::isa<cir::PointerType>(Load.getResult().getType())) {
        PointerInfo AddrInfo = Before.getPointerInfo(Load.getAddr());
        if (AddrInfo.State != PtrState::Unknown)
          Changed |= After->setPointerInfo(Load.getResult(), AddrInfo);
      }
    } else if (auto Cast = dyn_cast<cir::CastOp>(Op)) {
      // Propagate alias info through pointer casts (bitcast).
      if (mlir::isa<cir::PointerType>(Cast.getResult().getType()) &&
          mlir::isa<cir::PointerType>(Cast.getSrc().getType())) {
        PointerInfo Info = Before.getPointerInfo(Cast.getSrc());
        if (Info.State != PtrState::Unknown)
          Changed |= After->setPointerInfo(Cast.getResult(), Info);
      }
    }

    propagateIfChanged(After, Changed);
    return success();
  }

  void setToEntryState(LifetimeLattice *Lattice) override {
    propagateIfChanged(Lattice, Lattice->reset());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

void cir::diagnoseLifetimeSafety(ModuleOp Module, DiagnosticsEngine &Diags,
                                 SourceManager &SM) {
  // Clone module to avoid mutating the original (downstream passes need it
  // in structured form).
  OwningOpRef<ModuleOp> ClonedModule = Module.clone();

  // Flatten CIR and resolve gotos: converts structured regions to explicit
  // branches so DataFlowSolver can operate without crashing on
  // cir.break/cir.continue.
  mlir::PassManager PM(ClonedModule->getContext());
  PM.addPass(mlir::createCIRFlattenCFGPass());
  PM.addPass(mlir::createGotoSolverPass());
  if (failed(PM.run(*ClonedModule)))
    return;

  // Run analysis per function.
  ClonedModule->walk([&](cir::FuncOp FuncOp) {
    if (FuncOp.isDeclaration())
      return;

    DataFlowSolver Solver;
    loadBaselineAnalyses(Solver);
    Solver.load<LifetimeAnalysis>();
    if (failed(Solver.initializeAndRun(FuncOp)))
      return;

    // Post-solver walk: check return statements for pointers to local allocas.
    FuncOp.walk([&](cir::ReturnOp Ret) {
      if (!Ret.hasOperand())
        return;

      Value RetVal = Ret.getInput().front();
      if (!mlir::isa<cir::PointerType>(RetVal.getType()))
        return;

      const auto *State = Solver.lookupState<LifetimeLattice>(
          Solver.getProgramPointBefore(Ret.getOperation()));
      if (!State)
        return;

      PointerInfo Info = State->getPointerInfo(RetVal);
      if (Info.State != PtrState::PointsToLocal)
        return;

      // Get the alloca name for the diagnostic.
      StringRef VarName;
      if (auto Alloca = Info.TargetAlloca.getDefiningOp<cir::AllocaOp>())
        VarName = Alloca.getName();

      SourceLocation RetLoc = mlirLocToClangLoc(Ret.getLoc(), SM);
      if (RetLoc.isInvalid())
        return;

      // warn_ret_stack_addr_ref:
      //   arg0: 0=address-of, 1=reference-to
      //   arg1: variable name
      //   arg2: 0=local variable, 1=parameter, 2=compound literal
      //   arg3: 0=returned, 1=musttail
      Diags.Report(RetLoc, diag::warn_ret_stack_addr_ref)
          << /*address-of*/ 0 << VarName << /*local variable*/ 0
          << /*returned*/ 0;
    });
  });
}
