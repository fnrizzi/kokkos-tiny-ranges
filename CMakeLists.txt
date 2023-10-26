CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(pdakokkos CXX)

#=====================================================================
# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed.
    Please make a new directory (called a build directory) and run CMake from there.
    You may need to remove CMakeCache.txt.")
endif()

#=====================================================================
# check build type
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# convert cmake build type to lower string
string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if( NOT cmake_build_type_tolower STREQUAL "debug"
    AND NOT cmake_build_type_tolower STREQUAL "release")
  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\".
    Allowed values are Debug, Release (case-insensitive).")
endif()

#=====================================================================
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#include(CheckCXXCompilerFlag)
#check_cxx_compiler_flag("-std=c++20" COMPILER_SUPPORT_TARGET_CPP17)
#if(NOT COMPILER_SUPPORT_TARGET_CPP20)
#  message(FATAL_ERROR "C++ compiler does not support -std=c++20. This is required.")
#endif()
#MESSAGE(STATUS "Setting CMAKE_CXX_STANDARD=17")

#=====================================================================

find_package(Kokkos REQUIRED)
set(KOKKOS_TPLS Kokkos::kokkos)
set(PDA_ALL_TPLS_NEEDED ${KOKKOS_TPLS})

# try to find Gtest
find_package(GTest QUIET)
if (GTEST_FOUND)
  message(STATUS "\nGTest was found.")
  message(STATUS "GTEST_INCLUDE_DIRS = ${GTEST_INCLUDE_DIRS}")
  message(STATUS "GTEST_LIBRARIES = ${GTEST_LIBRARIES}\n")
else()
  message(STATUS "GTest not found, fetching it.")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()

enable_testing()
add_subdirectory(tests)
