// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=NEG

// Only the default (generic, 64-bit) address space gets a #cir.ptr_spec
// entry for now. AMDGPU's per-address-space pointer sizes (e.g. 32-bit
// private AS5) are not mirrored yet.

void foo() {}

// CHECK-DAG: dlti.dl_spec =
// CHECK-DAG:   !cir.ptr<!cir.void> = #cir.ptr_spec<size = 64, abi = 64, preferred = 64, index = 64>

// NEG-NOT: !cir.ptr<!cir.void, target_address_space
