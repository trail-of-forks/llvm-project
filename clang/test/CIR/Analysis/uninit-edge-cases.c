// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=uninit -Wuninitialized -verify %s -o /dev/null

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
