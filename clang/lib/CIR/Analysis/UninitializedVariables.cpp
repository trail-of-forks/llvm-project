//===- UninitializedVariables.cpp - CIR Uninit Variable Analysis ---*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Detects uses of uninitialized local variables using MLIR's
// DenseForwardDataFlowAnalysis on flattened CIR.
//
// Algorithm:
//   1. Clone the module (original must stay structured for downstream passes).
//   2. Flatten CIR via cir-flatten-cfg (converts structured regions to explicit
//      branches so DataFlowSolver can operate without crashing on
//      cir.break/cir.continue).
//   3. For each cir::FuncOp, run DataFlowSolver with DeadCodeAnalysis +
//      UninitAnalysis to compute per-alloca initialization state at each
//      program point.
//   4. Walk cir::LoadOp ops and emit diagnostics for loads from allocas that
//      are (possibly) uninitialized.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Analysis/UninitializedVariables.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
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
// InitializationLattice
//===----------------------------------------------------------------------===//

namespace {

/// Per-variable initialization state, matching the 2-bit lattice used by the
/// CFG-based UninitializedValues analysis.
enum class VarState : uint8_t {
  Uninitialized = 0,
  Initialized = 1,
  MayUninitialized = 2,
};

/// Dense lattice tracking initialization state for each cir.alloca in a
/// function. The map key is the SSA Value produced by the alloca; variables
/// not present in the map are treated as Uninitialized.
class InitializationLattice : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitializationLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const auto &other = static_cast<const InitializationLattice &>(rhs);
    ChangeResult changed = ChangeResult::NoChange;

    for (const auto &[val, otherState] : other.varStates) {
      auto it = varStates.find(val);
      if (it == varStates.end()) {
        // Not in our map means Uninitialized. Join with other's state.
        VarState joined = (otherState == VarState::Uninitialized)
                              ? VarState::Uninitialized
                              : VarState::MayUninitialized;
        varStates[val] = joined;
        if (joined != VarState::Uninitialized)
          changed = ChangeResult::Change;
      } else if (it->second != otherState) {
        // States differ at a merge point -> MayUninitialized.
        if (it->second != VarState::MayUninitialized) {
          it->second = VarState::MayUninitialized;
          changed = ChangeResult::Change;
        }
      }
    }

    return changed;
  }

  void print(raw_ostream &os) const override {
    os << "InitializationLattice{";
    bool first = true;
    for (const auto &[val, state] : varStates) {
      if (!first)
        os << ", ";
      first = false;
      os << val << "=";
      switch (state) {
      case VarState::Uninitialized:
        os << "Uninit";
        break;
      case VarState::Initialized:
        os << "Init";
        break;
      case VarState::MayUninitialized:
        os << "MayUninit";
        break;
      }
    }
    os << "}";
  }

  VarState getState(Value alloca) const {
    auto it = varStates.find(alloca);
    return it != varStates.end() ? it->second : VarState::Uninitialized;
  }

  ChangeResult setState(Value alloca, VarState state) {
    auto [it, inserted] = varStates.try_emplace(alloca, state);
    if (!inserted && it->second == state)
      return ChangeResult::NoChange;
    it->second = state;
    return ChangeResult::Change;
  }

  ChangeResult reset() {
    if (varStates.empty())
      return ChangeResult::NoChange;
    varStates.clear();
    return ChangeResult::Change;
  }

private:
  DenseMap<Value, VarState> varStates;
};

//===----------------------------------------------------------------------===//
// UninitAnalysis (DenseForwardDataFlowAnalysis)
//===----------------------------------------------------------------------===//

class UninitAnalysis
    : public DenseForwardDataFlowAnalysis<InitializationLattice> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               const InitializationLattice &before,
                               InitializationLattice *after) override {
    ChangeResult changed = after->join(before);

    if (auto store = dyn_cast<cir::StoreOp>(op)) {
      Value addr = store.getAddr();
      if (auto alloca = addr.getDefiningOp<cir::AllocaOp>())
        changed |= after->setState(alloca.getResult(), VarState::Initialized);
    }

    propagateIfChanged(after, changed);
    return success();
  }

  void setToEntryState(InitializationLattice *lattice) override {
    propagateIfChanged(lattice, lattice->reset());
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return true if this alloca should be excluded from uninitialized-variable
/// diagnostics.
static bool shouldSkipAlloca(cir::AllocaOp alloca) {
  // Variable was initialized at declaration (e.g. int x = 0;).
  if (alloca.getInit())
    return true;
  // Compiler-generated return value storage.
  if (alloca.getName() == "__retval")
    return true;
  return false;
}

} // namespace

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

void cir::diagnoseUninitializedVariables(ModuleOp Module,
                                         DiagnosticsEngine &Diags,
                                         SourceManager &SM) {
  // Clone module to avoid mutating the original (downstream passes need it
  // in structured form).
  OwningOpRef<ModuleOp> ClonedModule = Module.clone();

  // Flatten CIR: converts structured regions to explicit branches so
  // DataFlowSolver can operate correctly.
  mlir::PassManager PM(ClonedModule->getContext());
  PM.addPass(mlir::createCIRFlattenCFGPass());
  if (failed(PM.run(*ClonedModule)))
    return;

  // Run analysis per function.
  ClonedModule->walk([&](cir::FuncOp funcOp) {
    if (funcOp.isDeclaration())
      return;

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<UninitAnalysis>();
    if (failed(solver.initializeAndRun(funcOp)))
      return;

    // Walk loads and check initialization state.
    funcOp.walk([&](cir::LoadOp load) {
      Value addr = load.getAddr();
      auto alloca = addr.getDefiningOp<cir::AllocaOp>();
      if (!alloca)
        return;

      if (shouldSkipAlloca(alloca))
        return;

      const auto *state = solver.lookupState<InitializationLattice>(
          solver.getProgramPointBefore(load.getOperation()));
      if (!state)
        return;

      VarState varState = state->getState(alloca.getResult());
      if (varState == VarState::Initialized)
        return;

      SourceLocation UseLoc = mlirLocToClangLoc(load.getLoc(), SM);
      if (UseLoc.isInvalid())
        return;

      StringRef varName = alloca.getName();

      if (varState == VarState::Uninitialized) {
        // "variable %0 is uninitialized when used here"
        Diags.Report(UseLoc, diag::warn_uninit_var) << varName << 0;
      } else {
        // "variable %0 may be uninitialized when used here"
        Diags.Report(UseLoc, diag::warn_maybe_uninit_var) << varName << 0;
      }
    });
  });
}
