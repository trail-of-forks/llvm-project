; RUN: llc < %s -mtriple=armv7-none-eabi -verify-machineinstrs | FileCheck --check-prefixes=CT %s
; RUN: llc < %s -mtriple=armv6 -mattr=+ctselect -verify-machineinstrs | FileCheck --check-prefix=TEST-CT %s
; RUN: llc < %s -mtriple=armv6 -verify-machineinstrs | FileCheck --check-prefix=DEFAULT %s

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