// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm \
// RUN:   -fclangir-analysis=lifetime -Wreturn-stack-address \
// RUN:   -verify=expected,cir %s -o /dev/null
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:   -Wreturn-stack-address -verify=expected %s

// End-to-end test for CIR lifetime safety analysis with CIR-vs-Sema
// comparison. Lines with only 'cir-warning' are caught by CIR but missed
// by the baseline Sema AST check -- these demonstrate CIR's advantage.

// =====================================================================
// SECTION 1: Both Sema and CIR catch these (baseline equivalence)
// =====================================================================

// Direct return &local -- Sema's AST check catches the & operator.
int *test_direct() {
  int local = 42;
  return &local; // expected-warning {{address of stack memory associated with local variable 'local' returned}} \
                 // cir-warning {{address of stack memory associated with local variable local returned}}
}

// Conditional return &local -- Sema catches &local on the unsafe branch.
int *test_conditional(int cond) {
  int local = 42;
  if (cond)
    return &local; // expected-warning {{address of stack memory associated with local variable 'local' returned}} \
                   // cir-warning {{address of stack memory associated with local variable local returned}}
  return nullptr;
}

// =====================================================================
// SECTION 2: Only CIR catches these (CIR advantage over Sema AST check)
//
// Sema's checkReturnStackAddr only matches syntactic patterns like
// 'return &var'. CIR's DenseForwardDataFlowAnalysis tracks pointer-alias
// relationships through stores, loads, and casts, catching cases where
// the local address is obscured by intermediate variables.
// =====================================================================

// Via pointer variable -- the & doesn't appear in the return statement.
int *test_via_pointer() {
  int local = 42;
  int *p = &local;
  return p; // cir-warning {{address of stack memory associated with local variable local returned}}
}

// Via void* cast -- alias tracked through pointer cast.
int *test_void_cast() {
  int x = 1;
  void *v = &x;
  return (int *)v; // cir-warning {{address of stack memory associated with local variable x returned}}
}

// Multiple reassignments -- CIR tracks the final alias.
int *test_multiple_locals() {
  int a = 1;
  int b = 2;
  int *p = &a;
  p = &b;
  return p; // cir-warning {{address of stack memory associated with local variable b returned}}
}

// Inner scope dangling -- cir.scope structurally bounds the lifetime.
// After flattening, the inner alloca's PointsToLocal state propagates
// past the scope boundary, correctly detecting the dangling return.
int *test_inner_scope() {
  int *p;
  {
    int local = 42;
    p = &local;
  }
  return p; // cir-warning {{address of stack memory associated with local variable local returned}}
}

// Nested inner scopes -- alias propagates through multiple scope levels.
int *test_nested_scopes() {
  int *p;
  {
    {
      int local = 42;
      p = &local;
    }
  }
  return p; // cir-warning {{address of stack memory associated with local variable local returned}}
}

// For-loop body assigns &local -- safety-conservative join at loop exit
// detects that at least one iteration returns local memory.
int *test_for_loop() {
  int *p = nullptr;
  for (int i = 0; i < 1; i++) {
    int local = 42;
    p = &local;
  }
  return p; // cir-warning {{address of stack memory associated with local variable local returned}}
}

// While-loop body assigns &local.
int *test_while_loop(int n) {
  int *p = nullptr;
  int local = 42;
  while (n-- > 0) {
    p = &local;
  }
  return p; // cir-warning {{address of stack memory associated with local variable local returned}}
}

// Deeply nested if -- join correctly preserves PointsToLocal.
int *test_nested_if(int a, int b, int c) {
  int local = 42;
  int *p = nullptr;
  if (a) {
    if (b) {
      if (c) {
        p = &local;
      }
    }
  }
  return p; // cir-warning {{address of stack memory associated with local variable local returned}}
}

// Switch statement -- each case returning local address.
int *test_switch(int c) {
  int a = 1, b = 2;
  switch (c) {
  case 0:
    return &a; // expected-warning {{address of stack memory associated with local variable 'a' returned}} \
               // cir-warning {{address of stack memory associated with local variable a returned}}
  case 1:
    return &b; // expected-warning {{address of stack memory associated with local variable 'b' returned}} \
               // cir-warning {{address of stack memory associated with local variable b returned}}
  default:
    return nullptr;
  }
}

// Early return safe path, fall-through dangerous.
int *test_early_return(int *param, int cond) {
  if (cond)
    return param;
  int local = 42;
  return &local; // expected-warning {{address of stack memory associated with local variable 'local' returned}} \
                 // cir-warning {{address of stack memory associated with local variable local returned}}
}

// =====================================================================
// SECTION 3: True negatives -- neither Sema nor CIR should warn
// =====================================================================

int *test_return_param(int *p) {
  return p;
}

int test_return_value() {
  int x = 42;
  return x;
}

int test_deref_local() {
  int x = 42;
  int *p = &x;
  return *p;
}

int *test_return_nullptr() {
  int x = 42;
  (void)x;
  return nullptr;
}

int *test_param_with_inner_scope(int *param) {
  int *p = param;
  {
    int local = 42;
    (void)local;
  }
  return p;
}

int *test_reassign_to_param(int *param) {
  int local = 42;
  int *p = &local;
  p = param;
  return p;
}

int *test_store_param(int *param) {
  int *local_ptr = param;
  return local_ptr;
}

int *test_multi_param(int *a, int *b, int cond) {
  if (cond)
    return a;
  return b;
}

int test_load_local_value() {
  int x = 42;
  int *p = &x;
  int val = *p;
  return val;
}
