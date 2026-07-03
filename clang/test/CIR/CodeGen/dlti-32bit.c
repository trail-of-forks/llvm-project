// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// On a 32-bit target the default-address-space !cir.ptr entry is 32-bit.

void foo() {}

// CHECK-DAG: dlti.dl_spec =
// CHECK-DAG:   !cir.ptr<!cir.void> = #cir.ptr_spec<size = 32, abi = 32, preferred = 32, index = 32>
// CHECK-DAG:   !llvm.ptr = dense<32> : vector<4xi64>
