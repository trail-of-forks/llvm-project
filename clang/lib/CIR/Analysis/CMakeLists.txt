add_clang_library(clangCIRAnalysis
  CIRAnalysis.cpp
  DataFlowVerify.cpp
  MissingReturn.cpp
  SwitchFallthrough.cpp
  UninitializedVariables.cpp

  DEPENDS
  MLIRCIRPassIncGen
  MLIRCIROpsIncGen

  LINK_LIBS PUBLIC
  clangBasic

  MLIRAnalysis
  MLIRIR
  MLIRPass

  MLIRCIR
  MLIRCIRInterfaces
)
