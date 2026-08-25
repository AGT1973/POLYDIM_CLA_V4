cmake_minimum_required(VERSION 3.18)
project(einsof_cpp VERSION 40.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detectar cache line en compilacion (no hardcodeado)
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
  #include <new>
  int main() { return std::hardware_destructive_interference_size; }
" HAS_INTERFERENCE_SIZE)

add_library(einsof_cpp SHARED
    slerp_kernel.cpp
    tsqr.cpp
)

target_include_directories(einsof_cpp PUBLIC include)

# Habilitar SIMD si esta disponible
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" HAS_AVX2)
if(HAS_AVX2)
    target_compile_options(einsof_cpp PRIVATE -mavx2 -mfma)
endif()

# Tests
enable_testing()
add_executable(test_cpp tests/test_cpp.cpp)
target_link_libraries(test_cpp einsof_cpp)
add_test(NAME einsof_cpp_tests COMMAND test_cpp)
