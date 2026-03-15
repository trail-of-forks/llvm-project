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
//   3. For each cir::FuncOp, run DataFlowSolver with baseline analyses
//      (DeadCodeAnalysis + SparseConstantPropagation) and UninitAnalysis to
//      compute per-alloca initialization state at each program point.
//   4. Walk cir::LoadOp ops and emit diagnostics for loads from allocas that
//      are (possibly) uninitialized.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Analysis/UninitializedVariables.h"
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

    // If the other lattice is at bottom (entry/reset state), it carries no
    // information -- joining with bottom is the identity.
    if (other.atBottom)
      return ChangeResult::NoChange;

    // If this lattice is at bottom, adopt the other's state entirely.
    // This ensures the first predecessor's state is copied in, rather
    // than being spuriously merged with an empty "all Uninitialized" map.
    if (atBottom) {
      atBottom = false;
      varStates = other.varStates;
      return other.varStates.empty() ? ChangeResult::NoChange
                                     : ChangeResult::Change;
    }

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

    // Check keys in this but not in other (other treats absent as
    // Uninitialized). Without this second loop, the join is asymmetric:
    // a variable that is Initialized in this lattice but absent
    // (implicitly Uninitialized) in the other would not be demoted to
    // MayUninitialized, causing maybe-uninit patterns to go undetected.
    for (auto &[val, thisState] : varStates) {
      if (other.varStates.count(val))
        continue; // Already handled above.
      // thisState is Init or MayUninit, other is implicitly Uninit.
      if (thisState != VarState::MayUninitialized &&
          thisState != VarState::Uninitialized) {
        thisState = VarState::MayUninitialized;
        changed = ChangeResult::Change;
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
    atBottom = false;
    auto [it, inserted] = varStates.try_emplace(alloca, state);
    if (!inserted && it->second == state)
      return ChangeResult::NoChange;
    it->second = state;
    return ChangeResult::Change;
  }

  ChangeResult reset() {
    ChangeResult changed = ChangeResult::NoChange;
    if (!varStates.empty()) {
      varStates.clear();
      changed = ChangeResult::Change;
    }
    if (!atBottom) {
      atBottom = true;
      changed = ChangeResult::Change;
    }
    return changed;
  }

private:
  DenseMap<Value, VarState> varStates;
  /// True when this lattice is at "bottom" (the identity element for join).
  /// A bottom lattice has no information; joining with it copies the other
  /// side. This distinguishes the initial/reset state from a real state
  /// where all variables are Uninitialized.
  bool atBottom = true;
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

//===----------------------------------------------------------------------===//
// Complementary-condition suppression
//===----------------------------------------------------------------------===//

/// Return the underlying value of a boolean, stripping a single cir.unary(not).
/// Sets \p negated to true if a NOT was stripped.
static Value stripNot(Value boolVal, bool &negated) {
  negated = false;
  if (auto unary = boolVal.getDefiningOp<cir::UnaryOp>()) {
    if (unary.getKind() == cir::UnaryOpKind::Not) {
      negated = true;
      return unary.getInput();
    }
  }
  return boolVal;
}

/// Trace a boolean condition back to the alloca it was loaded from, if any.
/// Strips cir.cast(int_to_bool) and cir.load to find the underlying alloca.
/// Returns nullptr if the chain doesn't match the expected pattern.
static Value getConditionSource(Value cond) {
  // Strip int_to_bool cast.
  if (auto cast = cond.getDefiningOp<cir::CastOp>()) {
    if (cast.getKind() == cir::CastKind::int_to_bool)
      cond = cast.getSrc();
  }
  // Strip load to get the alloca.
  if (auto load = cond.getDefiningOp<cir::LoadOp>())
    return load.getAddr();
  return cond;
}

/// Return true if \p block contains a cir.store to \p alloca.
static bool blockStoresTo(Block *block, Value alloca) {
  for (Operation &op : *block) {
    if (auto store = dyn_cast<cir::StoreOp>(op)) {
      if (auto target = store.getAddr().getDefiningOp<cir::AllocaOp>())
        if (target.getResult() == alloca)
          return true;
    }
  }
  return false;
}

/// Collect brcond "diamond" patterns between two blocks. A diamond is a
/// brcond splitting into a true-branch (which may store to alloca) and a
/// false-branch, merging at a common successor.
///
/// Returns the condition base value and polarity for each diamond found.
struct DiamondInfo {
  Value condBase;
  bool negated;
  bool trueArmStores;
};

/// Walk backwards from \p end to \p limit collecting diamonds that branch on
/// conditions derived from the same loaded value. Stops at \p limit or when
/// no more single-predecessor blocks remain.
static void collectDiamonds(Block *end, Block *limit, Value alloca,
                            SmallVectorImpl<DiamondInfo> &diamonds) {
  Block *block = end;
  constexpr int maxSteps = 20; // Bound the walk to avoid pathological cases.
  for (int step = 0; step < maxSteps && block && block != limit; ++step) {
    // A diamond merge has exactly 2 predecessors.
    SmallVector<Block *, 4> preds(block->getPredecessors());
    if (preds.size() == 2) {
      // One predecessor should be a "true-arm" block that branches
      // unconditionally to this merge. The other should be the brcond block
      // itself (the false edge).
      for (int i = 0; i < 2; ++i) {
        Block *trueArm = preds[i];
        Block *condBlock = preds[1 - i];
        auto brcond = dyn_cast<cir::BrCondOp>(condBlock->getTerminator());
        if (!brcond)
          continue;
        // Verify the diamond shape: brcond true->trueArm, false->mergeBlock.
        if (brcond.getDestTrue() != trueArm || brcond.getDestFalse() != block)
          continue;
        bool neg;
        Value stripped = stripNot(brcond.getCond(), neg);
        Value base = getConditionSource(stripped);
        bool stores = blockStoresTo(trueArm, alloca);
        diamonds.push_back({base, neg, stores});
        // Continue walking from the brcond block.
        block = condBlock;
        goto nextStep;
      }
    }
    // Walk through single-predecessor blocks.
    if (preds.size() == 1) {
      block = preds[0];
      continue;
    }
    break;
  nextStep:;
  }
}

/// Check if a MayUninitialized result is a false positive caused by
/// complementary branch conditions that both initialize the variable.
///
/// Detects sequential diamond patterns:
///   if (cond)  x = ...;   // diamond 1: true-arm stores
///   if (!cond) x = ...;   // diamond 2: true-arm stores
///   use(x);               // MayUninitialized is a false positive
///
/// The algorithm walks backwards from the load collecting brcond diamonds.
/// If two diamonds branch on complementary conditions (same base value,
/// opposite polarity) and both true-arms store to the alloca, then the
/// variable is initialized on all feasible paths.
static bool isFalsePositiveDueToComplementaryBranches(cir::LoadOp load,
                                                      Value alloca) {
  SmallVector<DiamondInfo, 4> diamonds;
  collectDiamonds(load->getBlock(), nullptr, alloca, diamonds);

  // Look for a complementary pair among collected diamonds.
  for (size_t i = 0; i < diamonds.size(); ++i) {
    if (!diamonds[i].trueArmStores)
      continue;
    for (size_t j = i + 1; j < diamonds.size(); ++j) {
      if (!diamonds[j].trueArmStores)
        continue;
      // Same base condition, opposite polarity.
      if (diamonds[i].condBase == diamonds[j].condBase &&
          diamonds[i].negated != diamonds[j].negated)
        return true;
    }
  }
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

  // Flatten CIR and resolve gotos: converts structured regions to explicit
  // branches and replaces symbolic cir.goto/cir.label pairs with cir.br so
  // DataFlowSolver can see all CFG edges (including goto targets).
  mlir::PassManager PM(ClonedModule->getContext());
  PM.addPass(mlir::createCIRFlattenCFGPass());
  PM.addPass(mlir::createGotoSolverPass());
  if (failed(PM.run(*ClonedModule)))
    return;

  // Run analysis per function.
  ClonedModule->walk([&](cir::FuncOp funcOp) {
    if (funcOp.isDeclaration())
      return;

    DataFlowSolver solver;
    // Load baseline analyses: DeadCodeAnalysis needs
    // SparseConstantPropagation to resolve branch conditions and mark
    // successor blocks as live. Without it, conditional branches cause
    // successor blocks to remain dead and the analysis misses diagnostics.
    loadBaselineAnalyses(solver);
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

      // Suppress false positives from complementary branch conditions.
      // Example: if(cond) x=1; if(!cond) x=1; use(x);
      // The lattice reports MayUninitialized because it merges the two
      // if-blocks independently, but the conditions are exhaustive.
      if (varState == VarState::MayUninitialized &&
          isFalsePositiveDueToComplementaryBranches(load, alloca.getResult()))
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
