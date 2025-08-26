; RUN: llc < %s -mtriple=armv7-none-eabi -verify-machineinstrs | FileCheck --check-prefixes=CT %s
; RUN: llc < %s -mtriple=armv6 -verify-machineinstrs | FileCheck --check-prefix=DEFAULT %s

define i1 @ct_i1(i1 %cond, i1 %a, i1 %b) {
; CT-LABEL: ct_i1:
; CT: and
; CT: sub
; CT: rsb
; CT-NEXT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; DEFAULT: and
; DEFAULT: sub
; DEFAULT: rsb
; DEFAULT-NEXT: and
; DEFAULT-NEXT: and
; DEFAULT-NEXT: orr
; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call i1 @llvm.ct.select.i1(i1 %cond, i1 %a, i1 %b)
  ret i1 %sel
}

define i8 @ct_int8(i1 %cond, i8 %a, i8 %b) {
; CT-LABEL: ct_int8:
; CT: and
; CT: sub
; CT: rsb
; CT-NEXT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; DEFAULT: and
; DEFAULT: sub
; DEFAULT: rsb
; DEFAULT-NEXT: and
; DEFAULT-NEXT: and
; DEFAULT-NEXT: orr
; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call i8 @llvm.ct.select.i8(i1 %cond, i8 %a, i8 %b)
  ret i8 %sel
}

define i16 @ct_int16(i1 %cond, i16 %a, i16 %b) {
; CT-LABEL: ct_int16:
; CT: and
; CT: sub
; CT: rsb
; CT-NEXT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; DEFAULT: and
; DEFAULT: sub
; DEFAULT: rsb
; DEFAULT-NEXT: and
; DEFAULT-NEXT: and
; DEFAULT-NEXT: orr
; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call i16 @llvm.ct.select.i16(i1 %cond, i16 %a, i16 %b)
  ret i16 %sel
}

define i32 @ct_int32(i1 %cond, i32 %a, i32 %b) {
; CT-LABEL: ct_int32:
; CT: and
; CT: sub
; CT: rsb
; CT-NEXT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; DEFAULT: and
; DEFAULT: sub
; DEFAULT: rsb
; DEFAULT-NEXT: and
; DEFAULT-NEXT: and
; DEFAULT-NEXT: orr
; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call i32 @llvm.ct.select.i32(i1 %cond, i32 %a, i32 %b)
  ret i32 %sel
}

define i64 @ct_int64(i1 %cond, i64 %a, i64 %b) {
; CT-LABEL: ct_int64:
; CT: sub
; CT: rsb
; CT: and
; CT: and
; CT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call i64 @llvm.ct.select.i64(i1 %cond, i64 %a, i64 %b)
  ret i64 %sel
}

define float @ct_float(i1 %cond, float %a, float %b) {
; CT-LABEL: ct_float:
; CT: and
; CT: sub
; CT: rsb
; CT-NEXT: and
; CT-NEXT: and
; CT-NEXT: orr
; CT-NOT: b{{eq|ne|lt|gt|le|ge}}
; CT-NOT: j
; CT-NOT: mov
; CT-NOT: ldr

; DEFAULT-NOT: b{{eq|ne|lt|gt|le|ge}}
; DEFAULT-NOT: j
; DEFAULT-NOT: mov
entry:
  %sel = call float @llvm.ct.select.f32(i1 %cond, float %a, float %b)
  ret float %sel
}

define double @ct_f64(i1 %cond, double %a, double %b) {
; CT-LABEL: ct_f64:
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
  %sel = call double @llvm.ct.select.f64(i1 %cond, double %a, double %b)
  ret double %sel
}