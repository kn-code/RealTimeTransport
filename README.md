# RealTimeTransport

_RealTimeTransport_ is a C++20 library to simulate the real time dynamics of quantum transport processes.

## Building

The minimal steps to build and install the library are

```console
$ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
$ cd build
$ ninja install
```

A custom installation path can be chosen when configuring the project via

```console
$ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install
```

Tests can be compiled and run with

```console
$ ninja build_tests
$ ninja test
```

To compile the examples (and run one of them) use

```console
$ ninja build_examples
$ ./examples/double_dot_1
```

## Dependencies

This library uses some C++-20 features and therefore needs a not too old compiler. The following compilers were used successfully in the build process:

* gcc 11.4.0
* clang 16

Some functionalities require _blas_ and _lapacke_ packages, which are on Ubuntu-type systems most easily installed with

```console
$ apt install libopenblas-dev liblapacke-dev
```

## License

This library is licensed under the [Mozilla Public License (MPL) version 2.0](https://www.mozilla.org/en-US/MPL/2.0/FAQ/).