//===- DataFlowVerify.cpp - Verify DataFlowSolver on CIR --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs MLIR DataFlowSolver + DeadCodeAnalysis on CIR to empirically verify
// that cir.break and cir.continue terminators are correctly handled by the
// dataflow framework. This is a diagnostic pass used for infrastructure
// verification, not a user-facing analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CIRDATAFLOWVERIFY
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::dataflow;

namespace {

struct CIRDataFlowVerifyPass
    : public mlir::impl::CIRDataFlowVerifyBase<CIRDataFlowVerifyPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();

    if (failed(solver.initializeAndRun(op))) {
      op->emitError(
          "DataFlowSolver failed to initialize/run on CIR module");
      return signalPassFailure();
    }

    // Walk all blocks and report reachability status.
    op->walk([&](Operation *innerOp) {
      for (Region &region : innerOp->getRegions()) {
        for (Block &block : region) {
          auto *state = solver.lookupState<Executable>(
              solver.getProgramPointBefore(&block));

          // Determine if this block follows a cir.break or cir.continue.
          bool afterBreakOrContinue = false;
          llvm::StringRef terminatorName;

          // Check if any predecessor block ends with cir.break or
          // cir.continue. For blocks within structured CIR regions
          // (not CFG blocks), we check if the previous block in the
          // same region ends with one of these terminators.
          if (&block != &region.front()) {
            Block *prevBlock = block.getPrevNode();
            if (prevBlock && prevBlock->getTerminator()) {
              Operation *term = prevBlock->getTerminator();
              if (isa<cir::BreakOp>(term)) {
                afterBreakOrContinue = true;
                terminatorName = "cir.break";
              } else if (isa<cir::ContinueOp>(term)) {
                afterBreakOrContinue = true;
                terminatorName = "cir.continue";
              }
            }
          }

          bool isLive = state && state->isLive();
          const char *status = isLive ? "LIVE" : "DEAD";

          // Get a location for the remark: use first op in block, or
          // fall back to parent op.
          Operation *locOp =
              block.empty() ? innerOp : &block.front();

          if (afterBreakOrContinue) {
            locOp->emitRemark()
                << "block after " << terminatorName << " is "
                << status << " (dataflow verification)";
          } else {
            locOp->emitRemark() << "block is " << status;
          }
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createCIRDataFlowVerifyPass() {
  return std::make_unique<CIRDataFlowVerifyPass>();
}
