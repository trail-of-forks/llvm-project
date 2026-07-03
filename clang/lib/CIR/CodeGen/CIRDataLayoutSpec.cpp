//===--- CIRDataLayoutSpec.cpp - CIR data layout spec emission ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/CIRDataLayoutSpec.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/IR/DataLayout.h"

void cir::setCIRDataLayout(mlir::ModuleOp mod, const llvm::DataLayout &dl) {
  mlir::MLIRContext *ctx = mod.getContext();
  mlir::DataLayoutSpecInterface dlSpec = mlir::translateDataLayout(dl, ctx);

  // Mirror the `!llvm.ptr` (address space 0) pointer entry produced by the
  // importer into a `!cir.ptr<!cir.void>`-keyed #cir.ptr_spec entry, so
  // `!cir.ptr` layout queries resolve from the target data layout. Only the
  // default address space is supported for now; per-address-space entries
  // (`!llvm.ptr<N>`) are not mirrored yet. The `!llvm.ptr` entries are kept:
  // post-lowering consumers still need them.
  llvm::SmallVector<mlir::DataLayoutEntryInterface> entries(
      dlSpec.getEntries().begin(), dlSpec.getEntries().end());
  auto voidTy = cir::VoidType::get(ctx);
  for (mlir::DataLayoutEntryInterface entry : dlSpec.getEntries()) {
    auto ptrKey = llvm::dyn_cast_if_present<mlir::LLVM::LLVMPointerType>(
        llvm::dyn_cast_if_present<mlir::Type>(entry.getKey()));
    if (!ptrKey || ptrKey.getAddressSpace() != 0)
      continue;
    // Pointer entries are always [size, abi, preferred, index], in bits.
    auto values = mlir::cast<mlir::DenseIntElementsAttr>(entry.getValue())
                      .getValues<uint64_t>();
    assert(values.size() == 4 && "expected [size, abi, preferred, index]");
    entries.push_back(mlir::DataLayoutEntryAttr::get(
        cir::PointerType::get(voidTy),
        cir::PtrSpecAttr::get(ctx, static_cast<uint32_t>(values[0]),
                              static_cast<uint32_t>(values[1]),
                              static_cast<uint32_t>(values[2]),
                              static_cast<uint32_t>(values[3]))));
  }
  mod->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
               mlir::DataLayoutSpecAttr::get(ctx, entries));
}
