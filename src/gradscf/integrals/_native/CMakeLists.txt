cmake_minimum_required(VERSION 3.20)
project(gradscf_native LANGUAGES C CXX)

if(NOT JAX_INCLUDE_DIR)
  message(FATAL_ERROR "Set JAX_INCLUDE_DIR using jax.ffi.include_dir()")
endif()
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# The vendor tree is read-only source; generated headers stay in the build tree.
set(cint_VERSION 6.1.3)
set(cint_SOVERSION 6)
set(HAVE_EXPL 1)
set(HAVE_SQRTL 1)
set(HAVE_FABSL 1)
configure_file(vendor/libcint/include/cint.h.in generated/cint.h)
configure_file(vendor/libcint/src/cint_config.h.in generated/cint_config.h)
configure_file(include/config.h generated/config.h COPYONLY)

set(CINT_SOURCES
  cart2sph cint1e cint2e cint_bas fblas g1e g2e misc optimizer fmt
  rys_wheeler eigh rys_roots find_roots g1e_grids g3c1e g2c2e g3c2e)
list(TRANSFORM CINT_SOURCES PREPEND "vendor/libcint/src/")
list(TRANSFORM CINT_SOURCES APPEND ".c")
add_library(gradscf_integrals SHARED
  ${CINT_SOURCES} vendor/libcint/src/autocode/intor1.c
  vendor/pyscf/pyscf/lib/gto/fill_int2e.c csrc/ffi.cc)
target_include_directories(gradscf_integrals PRIVATE
  "${JAX_INCLUDE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/generated"
  vendor/libcint/src vendor/libcint/include)
target_link_libraries(gradscf_integrals PRIVATE m)
target_compile_options(gradscf_integrals PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-Wno-deprecated-non-prototype>)
# Link-local vendor symbols cannot interpose with a separately imported PySCF.
if(APPLE)
  target_link_options(gradscf_integrals PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.macos")
elseif(UNIX)
  target_link_options(gradscf_integrals PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    "LINKER:-z,defs")
else()
  message(FATAL_ERROR "Native integral build currently supports Linux and macOS")
endif()
if(GRADSCF_OUTPUT_DIR)
  set_target_properties(gradscf_integrals PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${GRADSCF_OUTPUT_DIR}")
endif()
