// 32-bit ARM uses a two-word array cookie { element_size, element_count }, so a
// new[] allocation is two size_t words larger than the element data: the data
// starts 8 bytes in, and delete[] recovers the allocation base 8 bytes before
// the data. (The generic Itanium cookie is a single size_t.)
//
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple arm-linux-gnueabihf -emit-llvm %s -o %t-ogcg.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-ogcg.ll %s

struct D {
  ~D();
  int v;
};

void make() {
  D *p = new D[10];
  delete[] p;
}

// 10 elements * 4 bytes + 8-byte cookie = 48.
// LLVM: call {{.*}}@_Znaj(i32 noundef 48)
// LLVM-DAG: store i32 4, ptr %{{[0-9]+}}
// LLVM-DAG: store i32 10, ptr %{{[0-9]+}}
// LLVM: getelementptr i8, ptr %{{[0-9]+}}, i32 8
// LLVM: getelementptr i8, ptr %{{[0-9]+}}, i32 -8

// OGCG: call {{.*}}@_Znaj(i32 noundef 48)
// OGCG-DAG: store i32 4, ptr
// OGCG-DAG: store i32 10, ptr
// OGCG: getelementptr inbounds i8, ptr %{{.*}}, i32 8
// OGCG: getelementptr inbounds i8, ptr %{{.*}}, i32 -8
