//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/BlockMatrices/BlockDiagonalCheb.h"

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <SciCore/ChebAdaptive.h>
#include <SciCore/Serialization.h>

namespace RealTimeTransport
{

struct BlockDiagonalCheb::Impl
{
    std::vector<SciCore::ChebAdaptive<MatrixType>> blocks; // Store each block of the matrix
};

BlockDiagonalCheb::BlockDiagonalCheb() : _pimpl{std::make_unique<Impl>()}
{
}

BlockDiagonalCheb::~BlockDiagonalCheb() = default;

BlockDiagonalCheb::BlockDiagonalCheb(BlockDiagonalCheb&& other) noexcept
    : _pimpl{std::make_unique<Impl>(std::move(*other._pimpl))}
{
}

BlockDiagonalCheb::BlockDiagonalCheb(const BlockDiagonalCheb& other) : _pimpl{std::make_unique<Impl>(*other._pimpl)}
{
}

BlockDiagonalCheb::BlockDiagonalCheb(std::vector<SciCore::ChebAdaptive<MatrixType>>&& blocks) noexcept
    : _pimpl{std::make_unique<Impl>(std::move(blocks))}
{
}

BlockDiagonalCheb::BlockDiagonalCheb(
    const std::function<MatrixType(int, SciCore::Real)>& f,
    int nBlocks,
    SciCore::Real a,
    SciCore::Real b,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    SciCore::Real hMin,
    bool* ok)
    : _pimpl{std::make_unique<Impl>()}
{
    using namespace SciCore;

    bool isOk = true;
    _pimpl->blocks.reserve(nBlocks);
    for (int i = 0; i < nBlocks; ++i)
    {
        auto f_i = [&f, i](Real t) -> MatrixType
        {
            return f(i, t);
        };

        bool isOk_i = false;
        _pimpl->blocks.push_back(ChebAdaptive(f_i, a, b, epsAbs, epsRel, hMin, &isOk_i));

        if (isOk_i == false)
        {
            isOk = isOk_i;
        }
    }

    if (ok != nullptr)
    {
        *ok = isOk;
    }
}

BlockDiagonalCheb::BlockDiagonalCheb(
    const std::function<MatrixType(int, SciCore::Real)>& f,
    int nBlocks,
    SciCore::Real a,
    SciCore::Real b,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    SciCore::Real hMin,
    tf::Executor& executor,
    bool* ok)
    : _pimpl{std::make_unique<Impl>()}
{
    using namespace SciCore;

    bool isOk = true;
    _pimpl->blocks.reserve(nBlocks);
    for (int i = 0; i < nBlocks; ++i)
    {
        auto f_i = [&f, i](Real t) -> MatrixType
        {
            return f(i, t);
        };

        bool isOk_i = false;
        _pimpl->blocks.push_back(ChebAdaptive(f_i, a, b, epsAbs, epsRel, hMin, executor, &isOk_i));

        if (isOk_i == false)
        {
            isOk = isOk_i;
        }
    }

    if (ok != nullptr)
    {
        *ok = isOk;
    }
}

BlockDiagonalCheb::BlockDiagonalCheb(
    const std::function<MatrixType(int, SciCore::Real)>& f,
    int nBlocks,
    const std::vector<SciCore::RealVector>& sections,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    SciCore::Real hMin,
    bool* ok)
    : _pimpl{std::make_unique<Impl>()}
{
    using namespace SciCore;

    if (static_cast<int>(sections.size()) != nBlocks)
    {
        throw Error("Invalid number of sections");
    }

    bool isOk = true;
    _pimpl->blocks.reserve(nBlocks);
    for (int i = 0; i < nBlocks; ++i)
    {
        auto f_i = [&f, i](Real t) -> MatrixType
        {
            return f(i, t);
        };

        bool isOk_i = false;
        _pimpl->blocks.push_back(ChebAdaptive(f_i, sections[i], epsAbs, epsRel, hMin, &isOk_i));

        if (isOk_i == false)
        {
            isOk = isOk_i;
        }
    }

    if (ok != nullptr)
    {
        *ok = isOk;
    }
}

BlockDiagonalCheb::BlockDiagonalCheb(
    const std::function<MatrixType(int, SciCore::Real)>& f,
    int nBlocks,
    const std::vector<SciCore::RealVector>& sections,
    SciCore::Real epsAbs,
    SciCore::Real epsRel,
    SciCore::Real hMin,
    tf::Executor& executor,
    bool* ok)
    : _pimpl{std::make_unique<Impl>()}
{
    using namespace SciCore;

    if (static_cast<int>(sections.size()) != nBlocks)
    {
        throw Error("Invalid number of sections");
    }

    bool isOk = true;
    _pimpl->blocks.reserve(nBlocks);
    for (int i = 0; i < nBlocks; ++i)
    {
        auto f_i = [&f, i](Real t) -> MatrixType
        {
            return f(i, t);
        };

        bool isOk_i = false;
        _pimpl->blocks.push_back(ChebAdaptive(f_i, sections[i], epsAbs, epsRel, hMin, executor, &isOk_i));

        if (isOk_i == false)
        {
            isOk = isOk_i;
        }
    }

    if (ok != nullptr)
    {
        *ok = isOk;
    }
}

BlockDiagonalCheb& BlockDiagonalCheb::operator=(const BlockDiagonalCheb& other)
{
    *_pimpl = *other._pimpl;
    return *this;
}

BlockDiagonalCheb& BlockDiagonalCheb::operator=(BlockDiagonalCheb&& other) noexcept
{
    *_pimpl = std::move(*other._pimpl);
    return *this;
}

bool BlockDiagonalCheb::operator==(const BlockDiagonalCheb& other) const
{
    return _pimpl->blocks == other._pimpl->blocks;
}

bool BlockDiagonalCheb::operator!=(const BlockDiagonalCheb& other) const
{
    return !operator==(other);
}

int BlockDiagonalCheb::numBlocks() const noexcept
{
    return static_cast<int>(_pimpl->blocks.size());
}

SciCore::Real BlockDiagonalCheb::lowerLimit() const
{
    if (_pimpl->blocks.size() == 0)
    {
        throw Error("Can't be used on empty object");
    }

    return _pimpl->blocks[0].lowerLimit();
}

SciCore::Real BlockDiagonalCheb::upperLimit() const
{
    if (_pimpl->blocks.size() == 0)
    {
        throw Error("Can't be used on empty object");
    }

    return _pimpl->blocks[0].upperLimit();
}

BlockDiagonalMatrix BlockDiagonalCheb::operator()(SciCore::Real t) const
{
    std::vector<MatrixType> blocks(_pimpl->blocks.size());
    for (size_t i = 0; i < _pimpl->blocks.size(); ++i)
    {
        blocks[i] = _pimpl->blocks[i](t);
    }

    return BlockDiagonalMatrix(std::move(blocks));
}

BlockDiagonalCheb::MatrixType BlockDiagonalCheb::operator()(int i, SciCore::Real t) const
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (i < 0 || i >= numBlocks())
    {
        throw Error("Invalid block index");
    }
#endif

    return _pimpl->blocks[i](t);
}

const SciCore::ChebAdaptive<BlockDiagonalCheb::MatrixType>& BlockDiagonalCheb::block(int i) const
{
#ifdef REAL_TIME_TRANSPORT_DEBUG
    if (i < 0 || i >= numBlocks())
    {
        throw Error("Invalid block index");
    }
#endif

    return _pimpl->blocks[i];
}

BlockDiagonalCheb BlockDiagonalCheb::diff() const
{
    BlockDiagonalCheb returnValue;
    returnValue._pimpl->blocks.reserve(_pimpl->blocks.size());

    for (const auto& c : _pimpl->blocks)
    {
        returnValue._pimpl->blocks.push_back(c.diff());
    }

    return returnValue;
}

BlockDiagonalCheb BlockDiagonalCheb::integrate() const
{
    BlockDiagonalCheb returnValue;
    returnValue._pimpl->blocks.reserve(_pimpl->blocks.size());

    for (const auto& c : _pimpl->blocks)
    {
        returnValue._pimpl->blocks.push_back(c.integrate());
    }

    return returnValue;
}

std::vector<SciCore::RealVector> BlockDiagonalCheb::sections() const
{
    using namespace SciCore;

    int nBlocks = numBlocks();

    if (nBlocks == 0)
    {
        throw Error("Method can't be used on an empty object");
    }

    std::vector<RealVector> returnValue(nBlocks);
    for (int i = 0; i < nBlocks; ++i)
    {
        returnValue[i] = _pimpl->blocks[i].sections();
    }

    return returnValue;
}

void BlockDiagonalCheb::serialize(cereal::BinaryInputArchive& archive)
{
    archive(_pimpl->blocks);
}

void BlockDiagonalCheb::serialize(cereal::BinaryOutputArchive& archive)
{
    archive(_pimpl->blocks);
}

void BlockDiagonalCheb::serialize(cereal::PortableBinaryInputArchive& archive)
{
    archive(_pimpl->blocks);
}

void BlockDiagonalCheb::serialize(cereal::PortableBinaryOutputArchive& archive)
{
    archive(_pimpl->blocks);
}

} // namespace RealTimeTransport
