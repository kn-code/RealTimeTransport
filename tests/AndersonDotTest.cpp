//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>

#include <RealTimeTransport/Models/AndersonDot.h>

using namespace SciCore;
using namespace RealTimeTransport;

TEST(AndersonDot, vectorizeAndOperatorize)
{
    RealVector T{
        {1, 0.4, 2.3}
    };
    RealVector mu{
        {0.9, 20, -15}
    };
    RealVector GammaDown{
        {1.1, 1, 0.9}
    };
    RealVector GammaUp{
        {0.2, 2.2, 3.4}
    };
    auto model = createModel<AndersonDot>(-0.9, 7.2, 10, T, mu, GammaDown, GammaUp);

    Model::OperatorType M       = Model::OperatorType::Random(4, 4);
    Model::SupervectorType Mvec = model->vectorize(M);
    Model::OperatorType M2      = model->operatorize(Mvec);
    EXPECT_EQ(M, M2);

    Model::SupervectorType V  = Model::SupervectorType::Random(16);
    Model::OperatorType Vop   = model->operatorize(V);
    Model::SupervectorType V2 = model->vectorize(Vop);
    EXPECT_EQ(V, V2);
}

TEST(AndersonDot, Serialization)
{
    Real epsilon = 0.5;
    Real B       = 3;
    Real U       = 4;
    RealVector T{
        {0.1, 0.9, 2.4}
    };
    RealVector mu{
        {-0.1, 0, 0.2}
    };
    RealVector GammaUp{
        {1, 1.1, 1.2}
    };
    RealVector GammaDown{
        {0.8, 0.7, 1.24}
    };

    auto dot = createModel<AndersonDot>(epsilon, B, U, T, mu, GammaUp, GammaDown);

    std::string archiveFilename = "anderson_dot_test_out.cereal";
    std::remove(archiveFilename.c_str());

    {
        std::ofstream os(archiveFilename, std::ios::binary);
        cereal::BinaryOutputArchive archive(os);
        archive(dot);
    }

    {
        //auto fromFile = createModel<AndersonDot>();
        std::unique_ptr<Model> fromFile;
        {
            std::ifstream is(archiveFilename, std::ios::binary);
            cereal::BinaryInputArchive archive(is);
            archive(fromFile);
        }

        EXPECT_EQ(dynamic_cast<AndersonDot&>(*dot), dynamic_cast<AndersonDot&>(*fromFile));
    }
}
