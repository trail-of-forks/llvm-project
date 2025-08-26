; RUN: llc < %s -mtriple=armv7-none-eabi -verify-machineinstrs | FileCheck --check-prefixes=CT %s
; RUN: llc < %s -mtriple=armv6 -verify-machineinstrs | FileCheck --check-prefix=DEFAULT %s

define <8 x i8> @ct_v8i8(i1 %cond, <8 x i8> %a, <8 x i8> %b) {
; CT-LABEL: ct_v8i8:
; CT: vand
; CT-NEXT: vldr
; CT-NEXT: vneg
; CT-NEXT: vbsl
; CT-NOT: ldr
; CT-NOT: vldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call <8 x i8> @llvm.ct.select.v8i8(i1 %cond, <8 x i8> %a, <8 x i8> %b)
  ret <8 x i8> %sel
}

define <4 x i16> @ct_v4i16(i1 %cond, <4 x i16> %a, <4 x i16> %b) {
; CT-LABEL: ct_v4i16:
; CT: vand
; CT-NEXT: vldr
; CT-NEXT: vneg
; CT-NEXT: vbsl
; CT-NOT: ldr
; CT-NOT: vldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call <4 x i16> @llvm.ct.select.v4i16(i1 %cond, <4 x i16> %a, <4 x i16> %b)
  ret <4 x i16> %sel
}

; Technically this should be handled the exact same as double.
define <2 x i32> @ct_v2i32(i1 %cond, <2 x i32> %a, <2 x i32> %b) {
; CT-LABEL: ct_v2i32:
; CT: vand
; CT-NEXT: vldr
; CT-NEXT: vneg
; CT-NEXT: vbsl
; CT-NOT: ldr
; CT-NOT: vldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call <2 x i32> @llvm.ct.select.v2i32(i1 %cond, <2 x i32> %a, <2 x i32> %b)
  ret <2 x i32> %sel
}

define <2 x float> @ct_v2f32(i1 %cond, <2 x float> %a, <2 x float> %b) {
; CT-LABEL: ct_v2f32:
; CT: vand
; CT-NEXT: vldr
; CT-NEXT: vneg
; CT-NEXT: vbsl
; CT-NOT: ldr
; CT-NOT: vldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call <2 x float> @llvm.ct.select.v2f32(i1 %cond, <2 x float> %a, <2 x float> %b)
  ret <2 x float> %sel
}

define <4 x float> @ct_v4f32(i1 %cond, <4 x float> %a, <4 x float> %b) {
; CT-LABEL: ct_v4f32:
; CT: vand
; CT: vldr
; CT: vneg
; CT: vbsl
; CT-NOT: ldr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j

; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call <4 x float> @llvm.ct.select.v4f32(i1 %cond, <4 x float> %a, <4 x float> %b)
  ret <4 x float> %sel
}