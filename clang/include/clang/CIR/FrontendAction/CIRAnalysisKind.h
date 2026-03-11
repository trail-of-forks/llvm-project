//===- CIRAnalysisKind.h - CIR analysis bitmask enum ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the CIRAnalysisKind bitmask enum used to select which CIR-based
// analyses to run. Each bit corresponds to a specific analysis pass.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_FRONTENDACTION_CIRANALYSISKIND_H
#define CLANG_CIR_FRONTENDACTION_CIRANALYSISKIND_H

namespace clang {

/// Bitmask enum for selecting CIR-based analysis passes.
///
/// Each entry corresponds to a CIR analysis that can replace or supplement
/// a traditional CFG-based warning. Flags are combined with bitwise OR
/// and tested with the \c has() helper.
enum class CIRAnalysisKind : unsigned {
  None = 0,
  SwitchFallthrough = 1 << 0,
  MissingReturn = 1 << 1,
  UninitVars = 1 << 2,
  Lifetime = 1 << 3,
  BufferOverflow = 1 << 4,
};

inline CIRAnalysisKind operator|(CIRAnalysisKind LHS, CIRAnalysisKind RHS) {
  return static_cast<CIRAnalysisKind>(static_cast<unsigned>(LHS) |
                                      static_cast<unsigned>(RHS));
}

inline CIRAnalysisKind &operator|=(CIRAnalysisKind &LHS,
                                   CIRAnalysisKind RHS) {
  LHS = LHS | RHS;
  return LHS;
}

inline CIRAnalysisKind operator&(CIRAnalysisKind LHS, CIRAnalysisKind RHS) {
  return static_cast<CIRAnalysisKind>(static_cast<unsigned>(LHS) &
                                      static_cast<unsigned>(RHS));
}

/// Return true if \p Set has the \p Kind bit(s) set.
inline bool has(CIRAnalysisKind Set, CIRAnalysisKind Kind) {
  return (Set & Kind) != CIRAnalysisKind::None;
}

} // namespace clang

#endif // CLANG_CIR_FRONTENDACTION_CIRANALYSISKIND_H
