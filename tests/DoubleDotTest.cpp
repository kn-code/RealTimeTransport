//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>

#include <RealTimeTransport/Models/DoubleDot.h>

using namespace SciCore;
using namespace RealTimeTransport;

TEST(DoubleDot, Serialization)
{
    Real epsilon1 = 0.5;
    Real epsilon2 = -0.5;
    Real U        = 4;
    Complex Omega(5, -7);
    RealVector T{
        {0.1, 0.9, 2.4}
    };
    RealVector mu{
        {-0.1, 0, 0.2}
    };
    RealVector Gamma1{
        {1, 1.1, 1.2}
    };
    RealVector Gamma2{
        {0.8, 0.7, 1.24}
    };

    auto model = createModel<DoubleDot>(epsilon1, epsilon2, U, Omega, T, mu, Gamma1, Gamma2);

    std::string archiveFilename = "double_dot_test_out.cereal";

    // Test binary archive
    std::remove(archiveFilename.c_str());
    {
        std::ofstream os(archiveFilename, std::ios::binary);
        cereal::BinaryOutputArchive archive(os);
        archive(model);
    }

    {
        std::unique_ptr<Model> fromFile;
        {
            std::ifstream is(archiveFilename, std::ios::binary);
            cereal::BinaryInputArchive archive(is);
            archive(fromFile);
        }

        EXPECT_EQ(dynamic_cast<DoubleDot&>(*model), dynamic_cast<DoubleDot&>(*fromFile));
    }

    // Test portable binary archive
    std::remove(archiveFilename.c_str());
    {
        std::ofstream os(archiveFilename, std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(os);
        archive(model);
    }

    {
        std::unique_ptr<Model> fromFile;
        {
            std::ifstream is(archiveFilename, std::ios::binary);
            cereal::PortableBinaryInputArchive archive(is);
            archive(fromFile);
        }

        EXPECT_EQ(dynamic_cast<DoubleDot&>(*model), dynamic_cast<DoubleDot&>(*fromFile));
    }
}
