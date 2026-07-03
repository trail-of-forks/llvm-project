// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// Subtracting pointers-to-pointers divides by the pointer size, which on
// i686 is 4 bytes. With the old hardcoded 64-bit !cir.ptr data layout the
// divisor came out as 8. Both ClangIR and classic codegen must agree.

// CHECK-LABEL: @ptr_ptr_diff(
// CHECK: sdiv exact i32 %{{.*}}, 4
long ptr_ptr_diff(void **a, void **b) { return a - b; }
