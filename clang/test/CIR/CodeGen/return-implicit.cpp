// RUN: %clang_cc1 -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test 1: Void function with no explicit return -- implicit return has
// is_implicit.
void implicit_void() {
  int x = 42;
}
// CHECK-LABEL: cir.func {{.*}}@_Z13implicit_voidv
// CHECK:         cir.return is_implicit

// Test 2: Explicit void return -- should NOT have is_implicit.
void explicit_void() {
  int x = 42;
  return;
}
// CHECK-LABEL: cir.func {{.*}}@_Z13explicit_voidv
// CHECK:         cir.return loc
// CHECK-NOT:     is_implicit

// Test 3: Non-void function with explicit return value -- no is_implicit.
int explicit_return() {
  return 42;
}
// CHECK-LABEL: cir.func {{.*}}@_Z15explicit_returnv
// CHECK:         cir.return %{{.*}} : !s32i loc
// CHECK-NOT:     is_implicit

// Test 4: main() has implicit return 0 in C++ -- is_implicit should appear.
int main() {
  int x = 1;
}
// CHECK-LABEL: cir.func {{.*}}@main
// CHECK:         cir.return %{{.*}} : !s32i is_implicit
