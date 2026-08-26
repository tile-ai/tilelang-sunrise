# TANG backend: source code generation and runtime-module integration.
file(GLOB TILE_LANG_TANG_ALWAYS_SRCS
  src/tang/target_utils.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_TANG_ALWAYS_SRCS})

if(NOT USE_TANG)
  return()
endif()

message(STATUS "TANG Backend is enabled")
file(GLOB TILE_LANG_TANG_ACTIVE_SRCS
  src/tang/codegen/codegen_tang.cc
  src/tang/codegen/rt_mod_tang.cc
  src/tang/op/*.cc
  src/tang/transform/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_TANG_ACTIVE_SRCS})
