// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// Test integer types
int test_int(int cond, int a, int b) {
  // CHECK-LABEL: define {{.*}} @test_int
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

long test_long(int cond, long a, long b) {
  // CHECK-LABEL: define {{.*}} @test_long
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i64 @llvm.ct.select.i64(i1 [[COND]], i64 %{{.*}}, i64 %{{.*}})
  // CHECK: ret i64 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

short test_short(int cond, short a, short b) {
  // CHECK-LABEL: define {{.*}} @test_short
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i16 @llvm.ct.select.i16(i1 [[COND]], i16 %{{.*}}, i16 %{{.*}})
  // CHECK: ret i16 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

unsigned char test_uchar(int cond, unsigned char a, unsigned char b) {
  // CHECK-LABEL: define {{.*}} @test_uchar
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i8 @llvm.ct.select.i8(i1 [[COND]], i8 %{{.*}}, i8 %{{.*}})
  // CHECK: ret i8 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

long long test_longlong(int cond, long long a, long long b) {
  // CHECK-LABEL: define {{.*}} @test_longlong
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i64 @llvm.ct.select.i64(i1 [[COND]], i64 %{{.*}}, i64 %{{.*}})
  // CHECK: ret i64 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test floating point types
float test_float(int cond, float a, float b) {
  // CHECK-LABEL: define {{.*}} @test_float
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call float @llvm.ct.select.f32(i1 [[COND]], float %{{.*}}, float %{{.*}})
  // CHECK: ret float [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

double test_double(int cond, double a, double b) {
  // CHECK-LABEL: define {{.*}} @test_double
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call double @llvm.ct.select.f64(i1 [[COND]], double %{{.*}}, double %{{.*}})
  // CHECK: ret double [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test pointer types
int *test_pointer(int cond, int *a, int *b) {
  // CHECK-LABEL: define {{.*}} @test_pointer
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call ptr @llvm.ct.select.p0(i1 [[COND]], ptr %{{.*}}, ptr %{{.*}})
  // CHECK: ret ptr [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test with different condition types
int test_char_cond(char cond, int a, int b) {
  // CHECK-LABEL: define {{.*}} @test_char_cond
  // CHECK: [[COND:%.*]] = icmp ne i8 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

int test_long_cond(long cond, int a, int b) {
  // CHECK-LABEL: define {{.*}} @test_long_cond
  // CHECK: [[COND:%.*]] = icmp ne i64 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test with boolean condition
int test_bool_cond(_Bool cond, int a, int b) {
  // CHECK-LABEL: define {{.*}} @test_bool_cond
  // CHECK: [[COND:%.*]] = trunc i8 %{{.*}} to i1
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test with constants
int test_constant_cond(void) {
  // CHECK-LABEL: define {{.*}} @test_constant_cond
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 true, i32 42, i32 24)
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(1, 42, 24);
}

int test_zero_cond(void) {
  // CHECK-LABEL: define {{.*}} @test_zero_cond
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 false, i32 42, i32 24)
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(0, 42, 24);
}

// Test type promotion
int test_promotion(int cond, short a, short b) {
  // CHECK-LABEL: define {{.*}} @test_promotion
  // CHECK-DAG: [[A_EXT:%.*]] = sext i16 %{{.*}} to i32
  // CHECK-DAG: [[B_EXT:%.*]] = sext i16 %{{.*}} to i32
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 [[A_EXT]], i32 [[B_EXT]])
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, (int)a, (int)b);
}

// Test mixed signedness
unsigned int test_mixed_signedness(int cond, int a, unsigned int b) {
  // CHECK-LABEL: define {{.*}} @test_mixed_signedness
  // CHECK-DAG: [[A_EXT:%.*]] = sext i32 %{{.*}} to i64
  // CHECK-DAG: [[B_EXT:%.*]] = zext i32 %{{.*}} to i64
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call i64 @llvm.ct.select.i64(i1 [[COND]], i64 [[A_EXT]], i64 [[B_EXT]])
  // CHECK: [[RESULT_TRUNC:%.*]] = trunc i64 [[RESULT]] to i32
  // CHECK: ret i32 [[RESULT_TRUNC]]
  return __builtin_ct_select(cond, (long)a, (long)b);
}

// Test complex expression
int test_complex_expr_alt(int x, int y) {
  // CHECK-LABEL: define {{.*}} @test_complex_expr_alt
  // CHECK-DAG: [[CMP:%.*]] = icmp sgt i32 %{{.*}}, 0
  // CHECK-DAG: [[ADD:%.*]] = add nsw i32 %{{.*}}, %{{.*}}
  // CHECK-DAG: [[SUB:%.*]] = sub nsw i32 %{{.*}}, %{{.*}}
  // Separate the final sequence to ensure proper ordering
  // CHECK-NEXT: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[CMP]], i32 [[ADD]], i32 [[SUB]])
  // CHECK-NEXT: ret i32 [[RESULT]]
  return __builtin_ct_select(x > 0, x + y, x - y);
}

// Test nested calls
int test_nested_structured(int cond1, int cond2, int a, int b, int c) {
  // CHECK-LABEL: define {{.*}} @test_nested_structured
  // Phase 1: Conditions (order doesn't matter)
  // CHECK-DAG: [[COND1:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[COND2:%.*]] = icmp ne i32 %{{.*}}, 0
  
  // Phase 2: Inner select (must happen before outer)
  // CHECK: [[INNER:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND2]], i32 %{{.*}}, i32 %{{.*}})
  
  // Phase 3: Outer select (must use inner result)
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND1]], i32 [[INNER]], i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond1, __builtin_ct_select(cond2, a, b), c);
}

// Test with function calls
int helper(int x) { return x * 2; }
int test_function_calls(int cond, int x, int y) {
  // CHECK-LABEL: define {{.*}} @test_function_calls
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[CALL1:%.*]] = call i32 @helper(i32 noundef %{{.*}})
  // CHECK-DAG: [[CALL2:%.*]] = call i32 @helper(i32 noundef %{{.*}})
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 [[CALL1]], i32 [[CALL2]])
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, helper(x), helper(y));
}
