#include "TargetInfo.h"
#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "CIRGenFunction.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenTypes.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace clang;
using namespace clang::CIRGen;

bool clang::CIRGen::isEmptyRecordForLayout(const ASTContext &context,
                                           QualType t) {
  const auto *rd = t->getAsRecordDecl();
  if (!rd)
    return false;

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *cxxrd = dyn_cast<CXXRecordDecl>(rd)) {
    if (cxxrd->isDynamicClass())
      return false;

    for (const auto &i : cxxrd->bases())
      if (!isEmptyRecordForLayout(context, i.getType()))
        return false;
  }

  for (const auto *i : rd->fields())
    if (!isEmptyFieldForLayout(context, i))
      return false;

  return true;
}

bool clang::CIRGen::isEmptyFieldForLayout(const ASTContext &context,
                                          const FieldDecl *fd) {
  if (fd->isZeroLengthBitField())
    return true;

  if (fd->isUnnamedBitField())
    return false;

  return isEmptyRecordForLayout(context, fd->getType());
}

namespace {

class X8664ABIInfo : public ABIInfo {
public:
  X8664ABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}
};

class X8664TargetCIRGenInfo : public TargetCIRGenInfo {
public:
  X8664TargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<X8664ABIInfo>(cgt)) {}
};

class ARMABIInfo : public ABIInfo {
public:
  ARMABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}

  cir::ABIArgInfo classifyReturnType(clang::CanQualType retTy) const override;
  cir::ABIArgInfo classifyArgumentType(clang::CanQualType argTy) const override;
};

class ARMTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  ARMTargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<ARMABIInfo>(cgt)) {}
};

cir::ABIArgInfo ARMABIInfo::classifyReturnType(clang::CanQualType retTy) const {
  // void -> Ignore.
  if (testIfIsVoidTy(retTy))
    return cir::ABIArgInfo::getIgnore();

  // Aggregates need transform-pass-level handling (sret / register-coerce);
  // mark NYI here so callers fail clearly rather than silently miscompiling.
  if (isAggregateTypeForABI(retTy)) {
    cgt.getCGModule().errorNYI(
        "ARM AAPCS: aggregate return classification is NYI in CodeGen");
    return cir::ABIArgInfo::getDirect();
  }

  // Small integer / bool / char / short / wchar / char8/16/32 are extended
  // to 32-bit per AAPCS. Signedness is taken from the source type.
  if (isPromotableIntegerTypeForABI(retTy)) {
    bool isSigned = retTy->hasSignedIntegerRepresentation();
    return cir::ABIArgInfo::getExtend(cgt.convertType(retTy), isSigned);
  }

  // Pointer / int32+ / float / SIMD: pass through.
  return cir::ABIArgInfo::getDirect();
}

cir::ABIArgInfo
ARMABIInfo::classifyArgumentType(clang::CanQualType argTy) const {
  if (isAggregateTypeForABI(argTy)) {
    cgt.getCGModule().errorNYI(
        "ARM AAPCS: aggregate argument classification is NYI in CodeGen");
    return cir::ABIArgInfo::getDirect();
  }

  if (isPromotableIntegerTypeForABI(argTy)) {
    bool isSigned = argTy->hasSignedIntegerRepresentation();
    return cir::ABIArgInfo::getExtend(cgt.convertType(argTy), isSigned);
  }

  return cir::ABIArgInfo::getDirect();
}

} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createX8664TargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<X8664TargetCIRGenInfo>(cgt);
}

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createARMTargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<ARMTargetCIRGenInfo>(cgt);
}

ABIInfo::~ABIInfo() noexcept = default;

// Default implementations: classify everything Direct, no coercion.
// Per-target ABIInfo subclasses override these to apply real rules.
cir::ABIArgInfo ABIInfo::classifyReturnType(clang::CanQualType /*retTy*/) const {
  return cir::ABIArgInfo::getDirect();
}

cir::ABIArgInfo
ABIInfo::classifyArgumentType(clang::CanQualType /*argTy*/) const {
  return cir::ABIArgInfo::getDirect();
}

void ABIInfo::computeInfo(CIRGenFunctionInfo &fi) const {
  fi.getReturnInfo() = classifyReturnType(fi.getReturnType());
  unsigned i = 0;
  for (const clang::CanQualType &argTy : fi.argTypes()) {
    fi.getArgInfo(i++) = classifyArgumentType(argTy);
  }
}

bool TargetCIRGenInfo::isNoProtoCallVariadic(
    const FunctionNoProtoType *fnType) const {
  // The following conventions are known to require this to be false:
  //   x86_stdcall
  //   MIPS
  // For everything else, we just prefer false unless we opt out.
  return false;
}

mlir::Value TargetCIRGenInfo::performAddrSpaceCast(
    CIRGenFunction &cgf, mlir::Value v, cir::TargetAddressSpaceAttr srcAddr,
    mlir::Type destTy, bool isNonNull) const {
  // Since target may map different address spaces in AST to the same address
  // space, an address space conversion may end up as a bitcast.
  if (cir::GlobalOp globalOp = v.getDefiningOp<cir::GlobalOp>())
    cgf.cgm.errorNYI("Global op addrspace cast");
  // Try to preserve the source's name to make IR more readable.
  return cgf.getBuilder().createAddrSpaceCast(v, destTy);
}
