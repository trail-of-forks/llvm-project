// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=lifetime -Wreturn-stack-address -verify %s -o /dev/null

// End-to-end test for CIR lifetime safety analysis.
// Verifies that -fclangir-analysis=lifetime produces -Wreturn-stack-address
// diagnostics via DenseForwardDataFlowAnalysis on flattened CIR.

// 1. Return address of local variable -- must warn.
// The Sema-level AST check catches direct &local; CIR analysis also catches it.
int *test_return_local_address() {
  int local = 42;
  return &local; // expected-warning {{address of stack memory associated with local variable 'local' returned}} \
                 // expected-warning {{address of stack memory associated with local variable}}
}

// 2. Return local address via intermediate pointer -- must warn.
// CIR catches this through pointer alias tracking; Sema AST check does not.
int *test_return_local_via_pointer() {
  int local = 42;
  int *p = &local;
  return p; // expected-warning {{address of stack memory associated with local variable}}
}

// 3. Return parameter pointer -- safe, no warning.
int *test_return_param_safe(int *p) {
  return p; // no warning -- returning parameter, not local address
}

// 4. Return value (not address) of local -- safe, no warning.
int test_no_return_no_warning() {
  int local = 42;
  int *p = &local;
  return *p; // no warning -- returning value, not address
}

// 5. Conditional return of local address -- must warn on the unsafe path.
// Both Sema AST check and CIR analysis catch this.
int *test_conditional_return(int cond) {
  int local = 42;
  if (cond)
    return &local; // expected-warning {{address of stack memory associated with local variable 'local' returned}} \
                   // expected-warning {{address of stack memory associated with local variable}}
  return nullptr;
}
