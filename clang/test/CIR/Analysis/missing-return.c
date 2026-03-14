// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=missing-return -Wreturn-type -verify %s -o /dev/null

// End-to-end test for A1 CIR missing-return analysis.
// Verifies that -fclangir-analysis=missing-return produces the same
// -Wreturn-type diagnostics as the CFG-based CheckFallThrough by inspecting
// cir.return is_implicit attribute directly.

// No return at all (AlwaysFallThrough).
int test_no_return(void) {
} // expected-warning {{non-void function does not return a value}}

// Some paths return, some don't (MaybeFallThrough).
int test_maybe_return(int x) {
  if (x)
    return 1;
} // expected-warning {{non-void function does not return a value in all control paths}}

// All paths return -- no warning.
int test_all_return(int x) {
  if (x)
    return 1;
  return 0;
}

// Void function -- no warning expected.
void test_void(void) {
}

// Declaration only (no body) -- no crash, no warning.
int test_declaration(void);

// Nested control flow -- return inside nested if.
int test_nested(int x, int y) {
  if (x) {
    if (y)
      return 1;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// === Algorithm Comparison: CIR vs CFG ===
//
// CIR approach (~30 lines):
//   Walk cir.func ops, check cir.return ops for is_implicit UnitAttr.
//   is_implicit is set by CIRGen when the compiler inserts an implicit return.
//   If present: warn. If explicit returns also exist: "in all control paths".
//
// CFG approach (~120 lines in CheckFallThrough, AnalysisBasedWarnings.cpp:552-680):
//   1. Build CFG from AST (AnalysisDeclContext::getCFG())
//   2. Compute reachability via ScanReachableFromBlock
//   3. Iterate exit block predecessors with FilterOptions
//   4. For each predecessor, scan backwards through CFGElements past destructors
//   5. Classify as HasLiveReturn / HasFakeEdge / HasPlainEdge / HasAbnormalEdge
//   6. Combine classifications into ControlFlowKind enum
//
// The is_implicit attribute encodes "compiler-inserted return" directly in IR,
// eliminating CFG construction and the predecessor scanning algorithm entirely.
