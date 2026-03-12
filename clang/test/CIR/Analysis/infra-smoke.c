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
// are not implemented. This test validates flag parsing, analysis dispatch
// infrastructure, and that the pipeline completes without crashes.

// A minimal non-void function that would trigger missing-return analysis.
int foo(int x) {
  if (x > 0)
    return x;
  // Missing return on this path -- no warning expected yet (analysis is a
  // stub), but the flag must be accepted without errors.
}

// A switch statement that would trigger switch-fallthrough analysis.
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
