//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/Model.h"
#include "RealTimeTransport/Error.h"
#include "RealTimeTransport/Utility.h"

namespace RealTimeTransport
{

Model::Model() noexcept
{
}

Model::~Model() noexcept
{
}

bool Model::operator==(const Model& other) const
{
    // If the derived types are the same then compare them
    if (typeid(*this) == typeid(other))
    {
        return isEqual(other);
    }
    else
    {
        return false;
    }
}

bool Model::operator!=(const Model& other) const
{
    return !operator==(other);
}

Model::OperatorType Model::P() const
{
    int dim = dimHilbertSpace();

    OperatorType returnValue = OperatorType::Identity(dim, dim);
    for (int l = 0; l < numStates(); ++l)
    {
        returnValue *= OperatorType::Identity(dim, dim) - 2.0 * d(l).adjoint() * d(l);
    }

    return returnValue;
}

Model::SupervectorType Model::vectorize(const Model::OperatorType& op) const
{
    using ConstVectorMap = Eigen::Map<const Eigen::Matrix<typename OperatorType::Scalar, Eigen::Dynamic, 1>>;

    return ConstVectorMap(op.data(), op.rows() * op.cols());
}

Model::OperatorType Model::operatorize(const Model::SupervectorType& supervector) const
{
    using ConstMatrixMap =
        Eigen::Map<const Eigen::Matrix<typename SupervectorType::Scalar, Eigen::Dynamic, Eigen::Dynamic>>;

    int dim     = supervector.size();
    int sqrtDim = std::sqrt(dim);

    if (sqrtDim * sqrtDim != dim)
    {
        throw Error("Invalid dimension");
    }

    return ConstMatrixMap(supervector.data(), sqrtDim, sqrtDim);
}

} // namespace RealTimeTransport
