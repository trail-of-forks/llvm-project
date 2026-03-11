// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=missing-return %s -o /dev/null 2>&1 | FileCheck %s --allow-empty
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=switch-fallthrough %s -o /dev/null 2>&1 | FileCheck %s --allow-empty
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=missing-return,switch-fallthrough %s -o /dev/null 2>&1 | FileCheck %s --allow-empty
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=bogus-analysis %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
//
// End-to-end infrastructure smoke test for CIR analysis pipeline.
// Verifies that:
//   1. -fclangir-analysis=missing-return is accepted and compilation succeeds
//   2. -fclangir-analysis=switch-fallthrough is accepted and compilation succeeds
//   3. Comma-separated analysis names are parsed correctly
//   4. Invalid analysis names produce a diagnostic error
//
// NOTE: No actual CIR analysis warnings are expected yet -- concrete analyses
// are implemented in Phases 11-12. This test validates flag parsing, analysis
// dispatch infrastructure, and that the pipeline completes without crashes.
//
// NOTE: INFRA-02 (diagnostic emission via DiagnosticsEngine::Report() with
// correct source locations) is validated as infrastructure (compiles, links)
// but not exercised observably in Phase 10. The first observable INFRA-02
// validation occurs in Phase 11 when switch-fallthrough analysis emits
// real warnings through mlirLocToClangLoc -> DiagnosticsEngine::Report().

// A minimal non-void function that would eventually trigger missing-return
// analysis once the CIR analysis is implemented in Phase 12.
int foo(int x) {
  if (x > 0)
    return x;
  // Missing return on this path -- no warning expected yet (analysis is a
  // stub until Phase 12), but the flag must be accepted without errors.
}

// A switch statement that would eventually trigger switch-fallthrough
// analysis once implemented in Phase 11.
void bar(int x) {
  switch (x) {
  case 1:
    x++;
    // Implicit fallthrough -- no warning expected yet
  case 2:
    break;
  }
}

// CHECK-NOT: error:
// CHECK-ERROR: invalid value 'bogus-analysis' in '-fclangir-analysis='
