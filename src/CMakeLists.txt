set(HEADER_LIST
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/ComputePropagator.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/Error.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/Model.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/Propagator.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/Utility.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/Models/AndersonDot.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/Models/DoubleDot.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/Models/ResonantLevel.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/BlockMatrices/BlockDiagonalCheb.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/BlockMatrices/BlockDiagonalMatrix.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/BlockMatrices/BlockHelper.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/BlockMatrices/BlockMatrix.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/BlockMatrices/BlockVector.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/BlockMatrices/MatrixExp.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/BlockMatrices/MatrixOperations.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/IteratedRG/CurrentKernel.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/IteratedRG/Diagrams.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/IteratedRG/MemoryKernel.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/RenormalizedPT/CurrentKernel.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/RenormalizedPT/Diagrams.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/RenormalizedPT/MemoryKernel.h"
    "${RealTimeTransport_SOURCE_DIR}/include/RealTimeTransport/extern/boost_unordered.hpp"
)

add_library(RealTimeTransport SHARED
    ComputePropagator.cpp
    Error.cpp
    Model.cpp
    Propagator.cpp
    Utility.cpp
    Models/AndersonDot.cpp
    Models/DoubleDot.cpp
    Models/ResonantLevel.cpp
    IteratedRG/CurrentKernel.cpp
    IteratedRG/Diagrams.cpp
    IteratedRG/MemoryKernel.cpp
    RenormalizedPT/CurrentKernel.cpp
    RenormalizedPT/Diagrams.cpp
    RenormalizedPT/MemoryKernel.cpp
    BlockMatrices/BlockDiagonalCheb.cpp
    BlockMatrices/BlockDiagonalMatrix.cpp
    BlockMatrices/BlockMatrix.cpp
    BlockMatrices/MatrixExp.cpp
    BlockMatrices/MatrixOperations.cpp
    BlockMatrices/BlockVector.cpp
    ${HEADER_LIST})
add_library(RealTimeTransport::RealTimeTransport ALIAS RealTimeTransport)

set_target_properties(RealTimeTransport PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

include(GenerateExportHeader)
generate_export_header(RealTimeTransport
    EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/../include/RealTimeTransport/RealTimeTransport_export.h
)

target_link_libraries(RealTimeTransport PUBLIC BLAS::BLAS LAPACK::LAPACK SciCore::SciCore)

target_include_directories(RealTimeTransport PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(RealTimeTransport PUBLIC cxx_std_20)
target_compile_definitions(RealTimeTransport PRIVATE ${REALTIMETRANSPORT_COMPILE_DEFINITIONS})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Every app/lib using RealTimeTransport needs to be build for the same architecture,
    # otherwise there can be segfaults using the Eigen library.
    # Therefore -march=native must be public.
    target_compile_options(RealTimeTransport PUBLIC
    $<$<CXX_COMPILER_ID:GNU>:-march=native>
    $<$<CXX_COMPILER_ID:Clang>:-march=native>
    )

    target_compile_options(RealTimeTransport PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-fno-math-errno -ffinite-math-only -fno-trapping-math -freciprocal-math -fno-signaling-nans -fcx-limited-range>
    $<$<CXX_COMPILER_ID:Clang>:-fno-math-errno -ffinite-math-only -fno-trapping-math -freciprocal-math -Wno-c99-extensions>
    )
endif()

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})
