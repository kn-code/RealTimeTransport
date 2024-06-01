# Testing library
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG v1.14.0
)

include(GoogleTest)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(AndersonDotTest AndersonDotTest.cpp)
target_link_libraries(AndersonDotTest PRIVATE RealTimeTransport GTest::gtest_main)
gtest_discover_tests(AndersonDotTest)

add_executable(BlockDiagonalChebTest BlockDiagonalChebTest.cpp)
target_link_libraries(BlockDiagonalChebTest PRIVATE RealTimeTransport GTest::gtest_main)
gtest_discover_tests(BlockDiagonalChebTest)

add_executable(BlockDiagonalMatrixTest BlockDiagonalMatrixTest.cpp)
target_link_libraries(BlockDiagonalMatrixTest PRIVATE RealTimeTransport BLAS::BLAS GTest::gtest_main)
gtest_discover_tests(BlockDiagonalMatrixTest)

add_executable(BlockMatrixTest BlockMatrixTest.cpp)
target_link_libraries(BlockMatrixTest PRIVATE RealTimeTransport BLAS::BLAS GTest::gtest_main)
gtest_discover_tests(BlockMatrixTest)

add_executable(BlockHelperTest BlockHelperTest.cpp)
target_link_libraries(BlockHelperTest PRIVATE RealTimeTransport GTest::gtest_main)
gtest_discover_tests(BlockHelperTest)

add_executable(BlockVectorTest BlockVectorTest.cpp)
target_link_libraries(BlockVectorTest PRIVATE RealTimeTransport GTest::gtest_main)
gtest_discover_tests(BlockVectorTest)

add_executable(DoubleDotTest DoubleDotTest.cpp)
target_link_libraries(DoubleDotTest PRIVATE RealTimeTransport GTest::gtest_main)
gtest_discover_tests(DoubleDotTest)

add_executable(MatrixExpTest MatrixExpTest.cpp)
target_link_libraries(MatrixExpTest PRIVATE RealTimeTransport GTest::gtest_main)
gtest_discover_tests(MatrixExpTest)

add_executable(MatrixOperationsTest MatrixOperationsTest.cpp)
target_link_libraries(MatrixOperationsTest PRIVATE RealTimeTransport BLAS::BLAS GTest::gtest_main)
gtest_discover_tests(MatrixOperationsTest)

add_executable(RenormalizedPTMemoryKernelTest RenormalizedPTMemoryKernelTest.cpp)
target_link_libraries(RenormalizedPTMemoryKernelTest PRIVATE RealTimeTransport GTest::gtest_main)
gtest_discover_tests(RenormalizedPTMemoryKernelTest)

add_executable(ResonantLevelTest ResonantLevelTest.cpp)
target_link_libraries(ResonantLevelTest PRIVATE RealTimeTransport GTest::gtest_main)
gtest_discover_tests(ResonantLevelTest)

add_executable(UtilityTest UtilityTest.cpp)
target_link_libraries(UtilityTest PRIVATE RealTimeTransport BLAS::BLAS GTest::gtest_main)
gtest_discover_tests(UtilityTest)

# Define custom target for building tests
add_custom_target(build_tests COMMENT "Build all the test executables.")
add_dependencies(build_tests AndersonDotTest)
add_dependencies(build_tests BlockDiagonalChebTest)
add_dependencies(build_tests BlockMatrixTest)
add_dependencies(build_tests BlockDiagonalMatrixTest)
add_dependencies(build_tests BlockHelperTest)
add_dependencies(build_tests BlockVectorTest)
add_dependencies(build_tests DoubleDotTest)
add_dependencies(build_tests MatrixExpTest)
add_dependencies(build_tests MatrixOperationsTest)
add_dependencies(build_tests RenormalizedPTMemoryKernelTest)
add_dependencies(build_tests ResonantLevelTest)
add_dependencies(build_tests UtilityTest)