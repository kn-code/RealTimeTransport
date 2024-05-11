cmake_minimum_required(VERSION 3.25)

#
# Tested on Linux with following commands:
# 
# g++:
# cmake -S . -B build -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_BUILD_TYPE=Release -G Ninja
#
# Clang:
# cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++-15 -DCMAKE_Fortran_COMPILER=gfortran-11 -DCMAKE_CXX_FLAGS="-stdlib=libc++" -DCMAKE_BUILD_TYPE=Release -G Ninja
#
# Useful flags: CMAKE_INSTALL_PREFIX, CMAKE_PREFIX_PATH, BLA_VENDOR
#

project(
    RealTimeTransport
    VERSION 0.9.0
    DESCRIPTION "Real-time simulations of quantum dot dynamics"
    HOMEPAGE_URL "https://github.com/kn-code/RealTimeTransport"
    LANGUAGES CXX Fortran
)

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

include(FetchContent)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Set these options only if we are the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_COLOR_DIAGNOSTICS ON)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

    if(NOT APPLE)
        set(CMAKE_INSTALL_RPATH $ORIGIN)
    endif()

    # Useful for debugging/benchmarking are also -g -fno-omit-frame-pointer
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wnon-virtual-dtor -pedantic -Wcast-align -Wunused -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion")
    endif ()

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    set(FETCHCONTENT_QUIET OFF)

    include(CTest)
endif()

#
# Dependencies
#

# BLAS and LAPACK
set(LAPACKE_STATIC ON)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(LAPACKE REQUIRED)
find_library(FORTRAN_LIB NAMES gfortran HINTS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})

# SciCore
find_package(SciCore 0.9.0 QUIET)

# If SciCore was not found use FetchContent to download it
if(NOT SciCore_FOUND)
    message(STATUS "SciCore not found. Fetching from Git repository...")

    FetchContent_Declare(
        SciCore
        GIT_REPOSITORY https://github.com/kn-code/SciCore
        GIT_TAG        main
    )

    FetchContent_MakeAvailable(SciCore)
endif()

#
# Build library and tests
#

add_subdirectory(src)

# Testing only available if this is the main app and BUILD_TESTING is defined
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME) AND BUILD_TESTING)
    add_subdirectory(tests EXCLUDE_FROM_ALL)
endif()

# Examples only available if this is the main app
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()

#
# Installation instructions
#

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(RealTimeTransport_INSTALL_CMAKEDIR
    ${CMAKE_INSTALL_LIBDIR}/cmake/RealTimeTransport
    CACHE STRING "Path to RealTimeTransport CMake files"
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    TARGETS RealTimeTransport
    EXPORT RealTimeTransportTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT   RealTimeTransport_Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT   RealTimeTransport_Runtime
        NAMELINK_COMPONENT RealTimeTransport_Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT RealTimeTransport_Development
)

install(EXPORT RealTimeTransportTargets
    DESTINATION ${RealTimeTransport_INSTALL_CMAKEDIR}
    NAMESPACE RealTimeTransport::
    FILE RealTimeTransportTargets.cmake
    COMPONENT RealTimeTransport_Development
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/RealTimeTransportConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/RealTimeTransportConfig.cmake
    INSTALL_DESTINATION ${RealTimeTransport_INSTALL_CMAKEDIR}
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/RealTimeTransportConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/RealTimeTransportConfigVersion.cmake
    DESTINATION ${RealTimeTransport_INSTALL_CMAKEDIR}
)

export(EXPORT RealTimeTransportTargets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/RealTimeTransportTargets.cmake"
       NAMESPACE RealTimeTransport::)
