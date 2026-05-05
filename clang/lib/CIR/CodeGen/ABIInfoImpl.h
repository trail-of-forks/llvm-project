//===- ABIInfoImpl.h - Helpers shared across CIR ABIInfo classifiers ----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic helpers used by per-target ABIInfo subclasses. The intent is that
// every helper here is target-agnostic: per-target rules are composed by
// calling these helpers with target-specific parameters.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_ABIINFOIMPL_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_ABIINFOIMPL_H

#include "clang/AST/CanonicalType.h"
#include "clang/AST/Type.h"

namespace clang::CIRGen {

/// Returns true if `t` is `void`. Convenience wrapper kept here so each
/// target's ABIInfo can read uniformly.
inline bool testIfIsVoidTy(clang::QualType t) {
  return t->isVoidType();
}

/// True for types that the ABI treats as aggregates: structs, unions, arrays,
/// complex types. C++ records and member-pointer types also count.
inline bool isAggregateTypeForABI(clang::QualType t) {
  return t->isRecordType() || t->isArrayType() || t->isAnyComplexType() ||
         t->isMemberFunctionPointerType();
}

/// True for small integer types that the ABI requires to be sign- or
/// zero-extended to the target word width: bool, char, short, and any enum
/// whose promotion type is one of those.
inline bool isPromotableIntegerTypeForABI(clang::QualType t) {
  if (const auto *bt = t->getAs<clang::BuiltinType>()) {
    switch (bt->getKind()) {
    case clang::BuiltinType::Bool:
    case clang::BuiltinType::Char_S:
    case clang::BuiltinType::Char_U:
    case clang::BuiltinType::SChar:
    case clang::BuiltinType::UChar:
    case clang::BuiltinType::Short:
    case clang::BuiltinType::UShort:
    case clang::BuiltinType::WChar_S:
    case clang::BuiltinType::WChar_U:
    case clang::BuiltinType::Char8:
    case clang::BuiltinType::Char16:
    case clang::BuiltinType::Char32:
      return true;
    default:
      return false;
    }
  }
  // Enums are promoted by their underlying type.
  if (const auto *et = t->getAs<clang::EnumType>())
    return isPromotableIntegerTypeForABI(et->getDecl()->getIntegerType());
  return false;
}

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_ABIINFOIMPL_H
