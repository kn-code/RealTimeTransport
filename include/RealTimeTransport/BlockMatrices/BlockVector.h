//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_VECTOR_H
#define REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_VECTOR_H

#include <vector>

#include "../extern/boost_unordered.hpp"

#include <SciCore/Definitions.h>

#include "../Error.h"
#include "../RealTimeTransport_export.h"

namespace RealTimeTransport
{

// A vector of matrices, but most vector elements are zero. In this sense the vector is "sparse".
class REALTIMETRANSPORT_EXPORT BlockVector
{
  public:
    using Scalar              = SciCore::Complex;
    using MatrixType          = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using UnorderedElementMap = boost::unordered_flat_map<int, MatrixType>;

    BlockVector() noexcept;
    BlockVector(UnorderedElementMap&& elements) noexcept;

    BlockVector& operator+=(const BlockVector& other);

    void reserve(size_t s);
    void emplace(int index, MatrixType&& A);

    UnorderedElementMap::iterator begin() noexcept;
    UnorderedElementMap::iterator end() noexcept;

    UnorderedElementMap::const_iterator begin() const noexcept;
    UnorderedElementMap::const_iterator end() const noexcept;

    int size() const noexcept;
    bool contains(int i) const;

    UnorderedElementMap::iterator find(int i);
    UnorderedElementMap::const_iterator find(int i) const;

    void eraseZeroes(SciCore::Real prec = Eigen::NumTraits<Scalar>::dummy_precision());
    void addToBlock(int i, MatrixType&& A);

  private:
    UnorderedElementMap _elements;
};

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_VECTOR_H
