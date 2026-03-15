// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=lifetime -Wreturn-stack-address -verify %s -o /dev/null

// Edge-case tests for CIR lifetime safety analysis.
// These test patterns where CIR's structural properties provide advantages
// over CFG-based analysis.

// CIR advantage: cir.scope structurally bounds the lifetime of 'local'.
// CFG relies on CFGLifetimeEnds synthetic elements which don't encode
// scope nesting depth. CIR's region structure makes the lifetime boundary
// a first-class IR property.
int *test_inner_scope_dangling() {
  int *p;
  {
    int local = 42;
    p = &local;
  }
  // 'local' is out of scope here. CIR's cir.scope region structurally
  // ended, making all inner allocas dead.
  return p; // expected-warning {{address of stack memory}}
}

// Pointer assigned in outer scope; inner scope doesn't affect it.
// No warning expected.
int *test_nested_scope_safe(int *param) {
  int *p = param;
  {
    int local = 42;
    (void)local;
  }
  return p; // no warning -- p points to param, not inner local
}
