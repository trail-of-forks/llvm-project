; RUN: llc < %s -mtriple=armv7-none-eabi -verify-machineinstrs | FileCheck --check-prefixes=CT %s
; RUN: llc < %s -mtriple=armv6 -verify-machineinstrs | FileCheck --check-prefix=DEFAULT %s

define i1 @ct_i1(i1 %cond, i1 %a, i1 %b) {
entry:
  %sel = call i1 @llvm.ct.select.i1(i1 %cond, i1 %a, i1 %b)
  ret i1 %sel
}

define i8 @ct_int8(i1 %cond, i8 %a, i8 %b) {
entry:
  %sel = call i8 @llvm.ct.select.i8(i1 %cond, i8 %a, i8 %b)
  ret i8 %sel
}

define i16 @ct_int16(i1 %cond, i16 %a, i16 %b) {
entry:
  %sel = call i16 @llvm.ct.select.i16(i1 %cond, i16 %a, i16 %b)
  ret i16 %sel
}

define i32 @ct_int32(i1 %cond, i32 %a, i32 %b) {
entry:
  %sel = call i32 @llvm.ct.select.i32(i1 %cond, i32 %a, i32 %b)
  ret i32 %sel
}

define i64 @ct_int64(i1 %cond, i64 %a, i64 %b) {
entry:
  %sel = call i64 @llvm.ct.select.i64(i1 %cond, i64 %a, i64 %b)
  ret i64 %sel
}

define half @ct_half(i1 %cond, half %a, half %b) {
entry:
  %sel = call half @llvm.ct.select.f16(i1 %cond, half %a, half %b)
  ret half %sel
}

define bfloat @ct_half(i1 %cond, bfloat %a, bfloat %b) {
entry:
  %sel = call bfloat @llvm.ct.select.bf16(i1 %cond, bfloat %a, bfloat %b)
  ret bfloat %sel
}

define float @ct_float(i1 %cond, float %a, float %b) {
entry:
  %sel = call float @llvm.ct.select.f32(i1 %cond, float %a, float %b)
  ret float %sel
}

define double @ct_f64(i1 %cond, double %a, double %b) {
entry:
  %sel = call double @llvm.ct.select.f64(i1 %cond, double %a, double %b)
  ret double %sel
}