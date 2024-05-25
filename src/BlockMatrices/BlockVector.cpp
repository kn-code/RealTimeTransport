//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/BlockMatrices/BlockVector.h"

namespace RealTimeTransport
{

BlockVector::BlockVector() noexcept
{
}

BlockVector::BlockVector(BlockVector::UnorderedElementMap&& elements) noexcept : _elements(std::move(elements))
{
}

BlockVector& BlockVector::operator+=(const BlockVector& other)
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

void BlockVector::reserve(size_t s)
{
    _elements.reserve(s);
}

void BlockVector::emplace(int index, BlockVector::MatrixType&& A)
{
    _elements.emplace(index, std::move(A));
}

BlockVector::UnorderedElementMap::iterator BlockVector::begin() noexcept
{
    return _elements.begin();
}

BlockVector::UnorderedElementMap::iterator BlockVector::end() noexcept
{
    return _elements.end();
}

BlockVector::UnorderedElementMap::const_iterator BlockVector::begin() const noexcept
{
    return _elements.begin();
}

BlockVector::UnorderedElementMap::const_iterator BlockVector::end() const noexcept
{
    return _elements.end();
}

int BlockVector::size() const noexcept
{
    return static_cast<int>(_elements.size());
}

bool BlockVector::contains(int i) const
{
    return _elements.contains(i);
}

BlockVector::UnorderedElementMap::iterator BlockVector::find(int i)
{
    return _elements.find(i);
}

BlockVector::UnorderedElementMap::const_iterator BlockVector::find(int i) const
{
    return _elements.find(i);
}

void BlockVector::eraseZeroes(SciCore::Real prec)
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

void BlockVector::addToBlock(int i, BlockVector::MatrixType&& A)
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

} // namespace RealTimeTransport
