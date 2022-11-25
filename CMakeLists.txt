cmake_minimum_required(VERSION 3.16...3.28)

# Set the project name and language
project(
  shuffle_project
  VERSION 0.0.1
  DESCRIPTION "ZCash Miner"
  HOMEPAGE_URL "%%myurl%%"
  LANGUAGES CXX C CUDA)

# Not ideal to use this global variable, but necessary to make sure
# that tooling and projects use the same version
set(CMAKE_CXX_STANDARD 20)

# strongly encouraged to enable this globally to avoid conflicts between
# -Wpedantic being enabled and -std=c++20 and -std=gnu++20 for example
# when compiling with PCH enabled
set(CMAKE_CXX_EXTENSIONS ON)

# add dependencies
include(cmake/CPM.cmake)


CPMAddPackage("gh:catchorg/Catch2@3.4.0")

CPMAddPackage(
  NAME Boost
  VERSION 1.81.0
  GITHUB_REPOSITORY "boostorg/boost"
  GIT_TAG "boost-1.81.0"
)

CPMAddPackage("gh:fmtlib/fmt#10.1.1")


# Adding the src:
add_subdirectory(src)
