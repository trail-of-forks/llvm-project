// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=uninit -Wuninitialized -Wconditional-uninitialized -verify %s -o /dev/null

// Edge-case tests for CIR uninitialized variable analysis, demonstrating
// how the CIR approach differs from the CFG approach.
//
// CIR vs CFG algorithm comparison:
// The CIR analysis operates on SSA-based alloca/store/load patterns on
// flattened CIR, which preserves the exact initialization relationship
// through control flow. The CFG analysis uses a bit-vector lattice indexed
// by VarDecl position and a 77-line DiagUninitUse() terminator switch to
// classify uses by branch context. CIR eliminates this switch entirely
// because the enclosing CIR op IS the branch context.
//
// Structural advantage: CIR's alloca/store/load tracking naturally handles
// scope boundaries without special VarDecl lifetime reasoning, because
// cir.store to an alloca is a direct SSA def visible across all subsequent
// basic blocks after flattening.

// Nested scope: variable declared outer, initialized in inner scope block.
// CIR tracks the cir.store to the alloca through scope boundaries naturally
// because both the alloca and the store target the same SSA value.
int test_nested_scope_init(void) {
  int x;
  {
    {
      x = 10;
    }
  }
  return x; // no warning -- CIR sees store flows to load
}

// Variable initialized in one scope, used in another.
int test_cross_scope(void) {
  int x;
  { x = 5; }
  int y;
  { y = x; }
  return y;
}

// Uninitialized variable used only through a nested scope.
int test_uninit_in_scope(void) {
  int x;
  {
    return x; // expected-warning {{variable x is uninitialized when used here}}
  }
}

// Multiple nested scopes, partial initialization.
int test_multi_scope(int cond) {
  int x;
  int y;
  {
    x = 1;
  }
  {
    y = x; // no warning, x was initialized above
  }
  return y;
}

// CIR advantage: variable initialized unconditionally in inner scope.
// CFG-based analysis requires the 77-line DiagUninitUse() terminator switch
// to classify uses by their branch context when scope blocks introduce
// extra CFG nodes. CIR's alloca/store/load SSA relationship preserves
// the initialization directly -- the store to the alloca is visible
// at the load point regardless of scope nesting, without needing to
// reason about terminator types or block predecessor chains.
//
// This demonstrates the structural advantage described in ANLYS-03: CIR's
// SSA-based tracking through cir.store/cir.load does not need the
// DiagUninitUse() terminator switch that CFG requires to classify uses by
// their branch context.
int test_cir_advantage_scope_init(int *arr, int n) {
  int result;
  {
    // Variable initialized unconditionally in inner scope.
    // CFG creates separate blocks for the scope entry/exit,
    // but CIR tracks the store->load relationship via SSA.
    result = arr[0];
  }
  return result; // no warning -- CIR sees store dominates load via SSA
}

// CIR correctly identifies maybe-uninit through nested scope boundaries.
// The variable is conditionally initialized in an inner scope; the CFG
// analysis requires careful block-predecessor reasoning to determine this
// is MayUninitialized, while CIR's lattice join at the merge point
// naturally produces MayUninitialized from the asymmetric paths.
int test_cir_maybe_uninit_nested(int cond) {
  int x;
  {
    if (cond) {
      x = 42;
    }
  }
  return x; // expected-warning {{variable x may be uninitialized when used here}}
}
