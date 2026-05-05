add_clang_library(MLIRCIRTargetLowering
  ABIInfo.cpp
  CIRCXXABI.cpp
  LowerModule.cpp
  LowerItaniumCXXABI.cpp
  LowerTypes.cpp
  TargetLoweringInfo.cpp

  DEPENDS
  clangBasic

  LINK_COMPONENTS
  TargetParser

  LINK_LIBS PUBLIC

  clangBasic
  MLIRIR
  MLIRPass
  MLIRDLTIDialect
  MLIRCIR
  MLIRCIRInterfaces
  MLIRLLVMDialect
)
