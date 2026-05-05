//==-- ABIArgInfo.h - Abstract info regarding ABI-specific arguments -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines ABIArgInfo and associated types used by CIR to track information
// regarding ABI-coerced types for function arguments and return values. This
// was moved to the common library as it might be used by both CIRGen and
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_ABIARGINFO_H
#define CLANG_CIR_ABIARGINFO_H

#include "mlir/IR/Types.h"
#include "clang/CIR/MissingFeatures.h"

namespace cir {

class ABIArgInfo {
public:
  enum Kind : uint8_t {
    /// Pass the argument directly using the normal converted CIR type,
    /// or by coercing to another specified type stored in 'CoerceToType'). If
    /// an offset is specified (in UIntData), then the argument passed is offset
    /// by some number of bytes in the memory representation. A dummy argument
    /// is emitted before the real argument if the specified type stored in
    /// "PaddingType" is not zero.
    Direct,

    /// Pass the argument after sign- or zero-extending it to a 32-bit (or
    /// larger) integer to satisfy the target ABI. Like Direct, but with
    /// the requirement that small integer types be widened.
    Extend,

    /// Pass the argument indirectly via a hidden pointer with the specified
    /// alignment (0 indicates default alignment) and address space.
    Indirect,

    /// Ignore the argument (treat as void). Useful for void and empty
    /// structs.
    Ignore,

    // TODO: more argument kinds (IndirectAliased, Expand, CoerceAndExpand,
    // InAlloca) will be added as the upstreaming proceeds.
  };

private:
  mlir::Type typeData;
  struct DirectAttrInfo {
    unsigned offset;
    unsigned align;
  };
  struct IndirectAttrInfo {
    unsigned align;
    unsigned addrSpace;
  };
  union {
    DirectAttrInfo directAttr;
    IndirectAttrInfo indirectAttr;
  };
  bool signExt : 1;
  Kind theKind;

public:
  ABIArgInfo(Kind k = Direct)
      : directAttr{0, 0}, signExt(false), theKind(k) {}

  static ABIArgInfo getDirect(mlir::Type ty = nullptr) {
    ABIArgInfo info(Direct);
    info.setCoerceToType(ty);
    assert(!cir::MissingFeatures::abiArgInfo());
    return info;
  }

  /// Sign- or zero-extend a small integer argument to an ABI-required width.
  /// `ty` carries the signedness information so this works whether the
  /// extension is sext or zext.
  static ABIArgInfo getExtend(mlir::Type ty, bool isSigned = true) {
    ABIArgInfo info(Extend);
    info.setCoerceToType(ty);
    info.signExt = isSigned;
    return info;
  }

  static ABIArgInfo getIndirect(unsigned align = 0, unsigned addrSpace = 0) {
    ABIArgInfo info(Indirect);
    info.indirectAttr = {align, addrSpace};
    return info;
  }

  /// Convenience: passes by reference with the natural alignment of the type.
  /// Alignment of 0 means "use the default for the target".
  static ABIArgInfo getNaturalAlignIndirect(mlir::Type /*ty*/ = nullptr,
                                            unsigned addrSpace = 0) {
    return getIndirect(/*align=*/0, addrSpace);
  }

  static ABIArgInfo getIgnore() { return ABIArgInfo(Ignore); }

  Kind getKind() const { return theKind; }
  bool isDirect() const { return theKind == Direct; }
  bool isExtend() const { return theKind == Extend; }
  bool isIndirect() const { return theKind == Indirect; }
  bool isIgnore() const { return theKind == Ignore; }

  bool isSignExt() const {
    assert(isExtend() && "Invalid kind!");
    return signExt;
  }

  bool canHaveCoerceToType() const {
    assert(!cir::MissingFeatures::abiArgInfo());
    return isDirect() || isExtend();
  }

  unsigned getDirectOffset() const {
    assert(!cir::MissingFeatures::abiArgInfo());
    return directAttr.offset;
  }

  unsigned getIndirectAlign() const {
    assert(isIndirect() && "Invalid kind!");
    return indirectAttr.align;
  }

  unsigned getIndirectAddrSpace() const {
    assert(isIndirect() && "Invalid kind!");
    return indirectAttr.addrSpace;
  }

  mlir::Type getCoerceToType() const {
    assert(canHaveCoerceToType() && "invalid kind!");
    return typeData;
  }

  void setCoerceToType(mlir::Type ty) {
    assert(canHaveCoerceToType() && "invalid kind!");
    typeData = ty;
  }
};

} // namespace cir

#endif // CLANG_CIR_ABIARGINFO_H
