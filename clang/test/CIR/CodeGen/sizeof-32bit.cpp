// RUN: %clang_cc1 -std=c++20 -triple nvptx-nvidia-cuda -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

unsigned sz = sizeof(int);

// CHECK: cir.global external @sz = #cir.int<4> : !u32i
