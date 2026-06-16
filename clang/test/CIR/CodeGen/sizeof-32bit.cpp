// RUN: %clang_cc1 -std=c++20 -triple nvptx-nvidia-cuda -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// On a target with 32-bit size_t, sizeof must materialize its constant in the
// AST result type (32-bit) rather than a hardcoded 64-bit integer type. The
// latter mismatched the APSInt width returned by EvaluateKnownConstInt and
// tripped IntAttr::get's verifier.

unsigned sz = sizeof(int);

// CHECK: cir.global external @sz = #cir.int<4> : !u32i
