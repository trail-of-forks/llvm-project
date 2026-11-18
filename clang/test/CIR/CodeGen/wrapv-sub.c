// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fwrapv -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fwrapv -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// With -fwrapv a signed subtraction wraps. It is not a saturating one.

int sub(int a, int b) { return a - b; }
// CIR-LABEL: cir.func{{.*}} @sub
// CIR: cir.binop(sub, %{{.+}}, %{{.+}}) : !s32i
// LLVM-LABEL: define{{.*}} i32 @sub
// LLVM: sub i32 %{{.+}}, %{{.+}}
// LLVM-NOT: llvm.ssub.sat

int add(int a, int b) { return a + b; }
// CIR-LABEL: cir.func{{.*}} @add
// CIR: cir.binop(add, %{{.+}}, %{{.+}}) : !s32i
// LLVM-LABEL: define{{.*}} i32 @add
// LLVM: add i32 %{{.+}}, %{{.+}}
