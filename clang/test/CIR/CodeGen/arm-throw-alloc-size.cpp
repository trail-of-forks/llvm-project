// The thrown_size argument to __cxa_allocate_exception is size_t, whose width
// is target-dependent.  On 32-bit ARM it must be i32, not the hardcoded i64
// used by 64-bit targets.
//
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=ARM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-x86.ll
// RUN: FileCheck --check-prefix=X86 --input-file=%t-x86.ll %s

void f() { throw 42; }

// ARM: declare ptr @__cxa_allocate_exception(i32)
// ARM: call ptr @__cxa_allocate_exception(i32 4)

// X86: declare ptr @__cxa_allocate_exception(i64)
// X86: call ptr @__cxa_allocate_exception(i64 4)
