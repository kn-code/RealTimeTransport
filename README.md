# RealTimeTransport

_RealTimeTransport_ is a C++20 library to simulate the real time dynamics of quantum transport processes. For a description of the implemented methods and usage examples, please see the [paper](
https://doi.org/10.1063/5.0220783
) and the [documentation](https://konstantin-nestmann.com/RealTimeTransport).

## Building the library

The minimal steps to build and install the library are

```bash
$ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
$ cd build
$ ninja install
```

A custom installation path can be chosen when configuring the project via

```bash
$ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install
```

Tests can be compiled and run with

```bash
$ ninja build_tests
$ ninja test
```

To compile the examples (and run one of them) use

```bash
$ ninja build_examples
$ ./examples/double_dot_1
```

## Consuming the library

After the library is installed, it is easiest to consume it via CMake. A minimal CMakeLists.txt file could be

```cmake
# Usage: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja -DCMAKE_PREFIX_PATH=/path/to/install/dir
cmake_minimum_required(VERSION 3.25)
project(MyProject)

find_package(RealTimeTransport REQUIRED)

add_executable(myExecutable main.cpp)
target_link_libraries(myExecutable PRIVATE RealTimeTransport::RealTimeTransport)
```

## Dependencies

This library requires CMake (version ≥ 3.25) to be built. Some C++-20 features are used which need a not too old compiler. The following compilers were used successfully in the build process:

* gcc 11.4.0
* clang 16

_BLAS_ and _LAPACKE_ can be enabled in the build process through the flags `-DEIGEN_USE_BLAS=ON` and `-DEIGEN_USE_LAPACKE=ON` (by default off). On Ubuntu-type systems the _BLAS_ and _LAPACKE_ libraries can be most easily installed with

```bash
$ apt install libopenblas-dev liblapacke-dev
```

## License

This library is licensed under the [Mozilla Public License (MPL) version 2.0](https://www.mozilla.org/en-US/MPL/2.0/FAQ/).