// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=uninit -Wuninitialized -Wconditional-uninitialized -verify %s -o /dev/null

// End-to-end test for CIR uninitialized variable analysis.
// Verifies that -fclangir-analysis=uninit produces -Wuninitialized
// diagnostics via DenseForwardDataFlowAnalysis on flattened CIR.

// 1. Always uninitialized -- must warn.
int test_always_uninit(void) {
  int x;
  return x; // expected-warning {{variable x is uninitialized when used here}}
}

// 2. Initialized variable -- no warning.
int test_initialized(void) {
  int x = 0;
  return x;
}

// 3. Initialized on all paths -- no warning.
int test_all_paths(int cond) {
  int x;
  if (cond)
    x = 1;
  else
    x = 2;
  return x;
}

// 4. Multiple variables, only one uninitialized.
int test_multiple_vars(void) {
  int x;
  int y = 1;
  return x + y; // expected-warning {{variable x is uninitialized when used here}}
}

// 5. Correct per-variable tracking: second var uninit, first is fine.
int test_per_var_tracking(void) {
  int x;
  int y;
  x = 1;
  return x + y; // expected-warning {{variable y is uninitialized when used here}}
}

// 6. Initialized inside nested scope block -- no warning.
int test_scope_init(void) {
  int x;
  {
    x = 42;
  }
  return x;
}

// 7. Conditionally initialized (if without else) -- maybe uninit.
int test_maybe_uninit(int cond) {
  int x;
  if (cond)
    x = 1;
  return x; // expected-warning {{variable x may be uninitialized when used here}}
}
