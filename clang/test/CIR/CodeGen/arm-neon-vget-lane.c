// The NEON lane-read intrinsics (vget_lane/vgetq_lane) lower to __builtin_neon_*
// builtins on 32-bit ARM (unlike AArch64, where arm_neon.h uses generic vector
// subscripting). Make sure CIR codegen lowers them to a vector element
// extraction instead of reporting them as unimplemented.

// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -target-feature +neon -ffreestanding -fclangir -emit-llvm %s -o - | FileCheck %s

#include <arm_neon.h>

// CHECK-LABEL: define dso_local i32 @get_s32(
// CHECK: extractelement <4 x i32> %{{.*}}, i32 2
int get_s32(int32x4_t v) { return vgetq_lane_s32(v, 2); }

// CHECK-LABEL: define dso_local float @get_f32(
// CHECK: extractelement <4 x float> %{{.*}}, i32 1
float get_f32(float32x4_t v) { return vgetq_lane_f32(v, 1); }

// CHECK-LABEL: define dso_local i16 @get_s16(
// CHECK: extractelement <4 x i16> %{{.*}}, i32 3
short get_s16(int16x4_t v) { return vget_lane_s16(v, 3); }
