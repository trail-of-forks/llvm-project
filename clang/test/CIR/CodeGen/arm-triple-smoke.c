// Phase 1 smoke test: ARM32 triple no longer rejected at CIR codegen time.
// Trivial scalar function lowers without crashing.
//
// RUN: %clang_cc1 -triple arm-linux-gnueabihf -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple armeb-linux-gnueabihf -fclangir -emit-cir %s -o %t-be.cir
// RUN: FileCheck --input-file=%t-be.cir %s --check-prefix=CIR

int add_one(int x) { return x + 1; }

// Pointer size must come from the data layout, not a hardcoded 64.
int *get_ptr(int *p) { return p + 1; }

// CIR: cir.func {{.*}}@add_one(%arg0: !s32i {{.*}}) -> !s32i
// CIR: cir.func {{.*}}@get_ptr(%arg0: !cir.ptr<!s32i>{{.*}}) -> !cir.ptr<!s32i>
