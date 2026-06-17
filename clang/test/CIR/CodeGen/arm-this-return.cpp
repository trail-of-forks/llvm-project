// 32-bit ARM returns 'this' from constructors and non-deleting destructors;
// other targets return void.
//
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=ARM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-x86.ll
// RUN: FileCheck --check-prefix=X86 --input-file=%t-x86.ll %s

struct S {
  S();
  ~S();
  int x;
};

S::S() { x = 1; }
S::~S() {}

// On ARM the constructor and destructor return the 'this' pointer, and the
// 'this' argument carries the 'returned' attribute.
// ARM: define{{.*}} ptr @_ZN1SC2Ev(ptr {{.*}} returned {{.*}})
// ARM: ret ptr
// ARM: define{{.*}} ptr @_ZN1SD2Ev(ptr {{.*}} returned {{.*}})
// ARM: ret ptr

// On x86_64 they return void and there is no 'returned' attribute.
// X86: define{{.*}} void @_ZN1SC2Ev(ptr
// X86-NOT: returned
// X86: define{{.*}} void @_ZN1SD2Ev(ptr
