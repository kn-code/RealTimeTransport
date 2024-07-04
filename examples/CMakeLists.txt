add_executable(conductance conductance.cpp)
target_link_libraries(conductance PRIVATE RealTimeTransport BLAS::BLAS)

add_executable(double_dot_1 double_dot_1.cpp)
target_link_libraries(double_dot_1 PRIVATE RealTimeTransport BLAS::BLAS)

add_executable(double_dot_2 double_dot_2.cpp)
target_link_libraries(double_dot_2 PRIVATE RealTimeTransport BLAS::BLAS)

add_executable(transient_anderson_dot transient_anderson_dot.cpp)
target_link_libraries(transient_anderson_dot PRIVATE RealTimeTransport BLAS::BLAS)

add_executable(two_loop two_loop.cpp)
target_link_libraries(two_loop PRIVATE RealTimeTransport BLAS::BLAS)

# Define custom target for building examples
add_custom_target(build_examples COMMENT "Build all examples.")
add_dependencies(build_examples conductance)
add_dependencies(build_examples double_dot_1)
add_dependencies(build_examples double_dot_2)
add_dependencies(build_examples transient_anderson_dot)
add_dependencies(build_examples two_loop)