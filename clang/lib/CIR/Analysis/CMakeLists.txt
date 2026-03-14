add_clang_library(clangCIRAnalysis
  CIRAnalysis.cpp
  DataFlowVerify.cpp
  SwitchFallthrough.cpp

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
