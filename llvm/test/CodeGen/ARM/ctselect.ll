; RUN: llc < %s -mtriple=armv7-none-eabi -verify-machineinstrs | FileCheck --check-prefixes=CT %s
; RUN: llc < %s -mtriple=armv6 -mattr=+ctselect -verify-machineinstrs | FileCheck --check-prefix=TEST-CT %s
; RUN: llc < %s -mtriple=armv6 -verify-machineinstrs | FileCheck --check-prefix=DEFAULT %s

define i8 @ct_int8(i1 %cond, i8 %a, i8 %b) {
; CT-LABEL: ct_int8:
; CT: and
; CT: and
; CT: orr
; CT-NOT: b{{eq|ne}}
; CT-NOT: j
; CT-NOT: {{mov|ldr}}
; TEST-CT: and
; TEST-CT: and
; TEST-CT: orr
; TEST-CT-NOT: b{{eq|ne}}
; TEST-CT-NOT: j
; TEST-CT-NOT: {{mov|ldr}}
; DEFAULT: {{mov|ldr}}
entry:
  %sel = call i8 @llvm.ct.select.i8(i1 %cond, i8 %a, i8 %b)
  ret i8 %sel
}

define i16 @ct_int16(i1 %cond, i16 %a, i16 %b) {
; CT-LABEL: ct_int16:
; CT: and
; CT: and
; CT: orr
; CT-NOT: b{{eq|ne}}
; CT-NOT: j
; CT-NOT: {{mov|ldr}}
; TEST-CT: and
; TEST-CT: and
; TEST-CT: orr
; TEST-CT-NOT: b{{eq|ne}}
; TEST-CT-NOT: j
; TEST-CT-NOT: {{mov|ldr}}
; DEFAULT: {{mov|ldr}}
entry:
  %sel = call i16 @llvm.ct.select.i16(i1 %cond, i16 %a, i16 %b)
  ret i16 %sel
}

define i32 @ct_int32(i1 %cond, i32 %a, i32 %b) {
; CT-LABEL: ct_int32:
; CT: and
; CT: and
; CT: orr
; CT-NOT: b{{eq|ne}}
; CT-NOT: j
; CT-NOT: {{mov|ldr}}
; TEST-CT: and
; TEST-CT: and
; TEST-CT: orr
; TEST-CT-NOT: b{{eq|ne}}
; TEST-CT-NOT: j
; TEST-CT-NOT: {{mov|ldr}}
; DEFAULT: {{mov|ldr}}
entry:
  %sel = call i32 @llvm.ct.select.i32(i1 %cond, i32 %a, i32 %b)
  ret i32 %sel
}

define i64 @ct_int64(i1 %cond, i64 %a, i64 %b) {
; CT-LABEL: ct_int64:
; CT: and
; CT: and
; CT: orr
; CT-NOT: b{{eq|ne}}
; CT-NOT: j
; CT-NOT: {{mov|ldr}}
; TEST-CT: and
; TEST-CT: and
; TEST-CT: orr
; TEST-CT-NOT: b{{eq|ne}}
; TEST-CT-NOT: j
; TEST-CT-NOT: {{mov|ldr}}
; DEFAULT: {{mov|ldr}}
entry:
  %sel = call i64 @llvm.ct.select.i64(i1 %cond, i64 %a, i64 %b)
  ret i64 %sel
}