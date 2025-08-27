; RUN: llc < %s -mtriple=armv7-none-eabi -verify-machineinstrs | FileCheck --check-prefixes=CT %s
; RUN: llc < %s -mtriple=armv6 -verify-machineinstrs | FileCheck --check-prefix=DEFAULT %s

define <8 x i8> @ct_v8i8(i1 %cond, <8 x i8> %a, <8 x i8> %b) {
entry:
  %sel = call <8 x i8> @llvm.ct.select.v8i8(i1 %cond, <8 x i8> %a, <8 x i8> %b)
  ret <8 x i8> %sel
}

define <4 x i16> @ct_v4i16(i1 %cond, <4 x i16> %a, <4 x i16> %b) {
entry:
  %sel = call <4 x i16> @llvm.ct.select.v4i16(i1 %cond, <4 x i16> %a, <4 x i16> %b)
  ret <4 x i16> %sel
}

define <2 x i32> @ct_v2i32(i1 %cond, <2 x i32> %a, <2 x i32> %b) {
entry:
  %sel = call <2 x i32> @llvm.ct.select.v2i32(i1 %cond, <2 x i32> %a, <2 x i32> %b)
  ret <2 x i32> %sel
}

define <1 x i64> @ct_v1i64(i1 %cond, <1 x i64> %a, <1 x i64> %b) {
entry:
  %sel = call <1 x i64> @llvm.ct.select.v1i64(i1 %cond, <1 x i64> %a, <1 x i64> %b)
  ret <1 x i64> %sel
}

define <2 x float> @ct_v2f32(i1 %cond, <2 x float> %a, <2 x float> %b) {
entry:
  %sel = call <2 x float> @llvm.ct.select.v2f32(i1 %cond, <2 x float> %a, <2 x float> %b)
  ret <2 x float> %sel
}

define <4 x half> @ct_v4f16(i1 %cond, <4 x half> %a, <4 x half> %b) {
entry:
  %sel = call <4 x half> @llvm.ct.select.v4f16(i1 %cond, <4 x half> %a, <4 x half> %b)
  ret <4 x half> %sel
}

define <4 x bfloat> @ct_v4bf16(i1 %cond, <4 x bfloat> %a, <4 x bfloat> %b) {
entry:
  %sel = call <4 x bfloat> @llvm.ct.select.v4bf16(i1 %cond, <4 x bfloat> %a, <4 x bfloat> %b)
  ret <4 x bfloat> %sel
}

def <16 x i8> @ct_v16i8(i1 %cond, <16 x i8> %a, <16 x i8> %b) {
entry:
  %sel = call <16 x i8> @llvm.ct.select.v16i8(i1 %cond, <16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %sel
}

def <8 x i16> @ct_v8i16(i1 %cond, <8 x i16> %a, <8 x i16> %b) {
entry:
  %sel = call <8 x i16> @llvm.ct.select.v8i16(i1 %cond, <8 x i16> %a, <8 x i16> %b)
  ret <8 x i16> %sel
}

def <4 x i32> @ct_v4i32(i1 %cond, <4 x i32> %a, <4 x i32> %b) {
entry:
  %sel = call <4 x i32> @llvm.ct.select.v4i32(i1 %cond, <4 x i32> %a, <4 x i32> %b)
}

def <2 x i64> @ct_v2i64(i1 %cond, <2 x i64> %a, <2 x i64> %b) {
entry:
  %sel = call <2 x i64> @llvm.ct.select.v2i64(i1 %cond, <2 x i64> %a, <2 x i64> %b)
}

define <4 x float> @ct_v4f32(i1 %cond, <4 x float> %a, <4 x float> %b) {
entry:
  %sel = call <4 x float> @llvm.ct.select.v4f32(i1 %cond, <4 x float> %a, <4 x float> %b)
  ret <4 x float> %sel
}

def <2 x double> @ct_v2f64(i1 %cond, <2 x double> %a, <2 x double> %b) {
entry:
  %sel = call <2 x double> @llvm.ct.select.v2f64(i1 %cond, <2 x double> %a, <2 x double> %b)
}

def <8 x half> @ct_v8f16(i1 %cond, <8 x half> %a, <8 x half> %b) {
entry:
  %sel = call <8 x half> @llvm.ct.select.v8f16(i1 %cond, <8 x half> %a, <8 x half> %b)
}

def <8 x bfloat> @ct_v8bf16(i1 %cond, <8 x bfloat> %a, <8 x bfloat> %b) {
entry:
  %sel = call <8 x bfloat> @llvm.ct.select.v8bf16(i1 %cond, <8 x bfloat> %a, <8 x bfloat> %b)
}