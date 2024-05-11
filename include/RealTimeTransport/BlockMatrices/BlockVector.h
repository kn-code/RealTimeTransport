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
template <typename T>
class REALTIMETRANSPORT_EXPORT BlockVector
{
  public:
    using MatrixType          = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Scalar              = T;
    using UnorderedElementMap = boost::unordered_flat_map<int, MatrixType>;

    BlockVector() noexcept
    {
    }

    BlockVector(UnorderedElementMap&& elements) noexcept : _elements(std::move(elements))
    {
    }

    BlockVector& operator+=(const BlockVector& other)
    {
        for (const auto& pair : other._elements)
        {
            int row = pair.first;
            auto it = _elements.find(row);
            if (it == _elements.end())
            {
                _elements[row] = pair.second;
            }
            else
            {
                it->second += pair.second;
            }
        }

        return *this;
    }

    void reserve(size_t s)
    {
        _elements.reserve(s);
    }

    void emplace(int index, MatrixType&& A)
    {
        _elements.emplace(index, std::move(A));
    }

    auto begin() noexcept
    {
        return _elements.begin();
    }

    auto end() noexcept
    {
        return _elements.end();
    }

    auto begin() const noexcept
    {
        return _elements.begin();
    }

    auto end() const noexcept
    {
        return _elements.end();
    }

    int size() const noexcept
    {
        return static_cast<int>(_elements.size());
    }

    bool contains(int i) const
    {
        return _elements.contains(i);
    }

    auto find(int i)
    {
        return _elements.find(i);
    }

    auto find(int i) const
    {
        return _elements.find(i);
    }

    void eraseZeroes(SciCore::Real prec = Eigen::NumTraits<Scalar>::dummy_precision())
    {
        std::vector<int> keysToRemove;
        for (const auto& pair : _elements)
        {
            if (pair.second.isZero(prec) == true)
            {
                keysToRemove.push_back(pair.first);
            }
        }

        for (const auto& key : keysToRemove)
        {
            _elements.erase(key);
        }
    }

    void addToBlock(int i, MatrixType&& A)
    {
        auto it = _elements.find(i);
        if (it == _elements.end())
        {
            _elements[i] = std::move(A);
        }
        else
        {
            it->second += A;
        }
    }

  private:
    UnorderedElementMap _elements;
};

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_BLOCK_MATRICES_BLOCK_VECTOR_H
