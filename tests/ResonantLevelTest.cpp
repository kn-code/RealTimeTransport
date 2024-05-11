//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#define REAL_TIME_TRANSPORT_DEBUG

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>

#include <RealTimeTransport/Models/ResonantLevel.h>

using namespace SciCore;
using namespace RealTimeTransport;

TEST(ResonantLevel, Serialization)
{
    Real epsilon = 0.5;
    RealVector T{
        {0.1, 0.9, 2.4}
    };
    RealVector mu{
        {-0.1, 0, 0.2}
    };
    RealVector Gamma{
        {1, 1.1, 1.2}
    };

    auto rlm = createModel<ResonantLevel>(epsilon, T, mu, Gamma);

    std::string archiveFilename = "rlm_test_out.cereal";
    std::remove(archiveFilename.c_str());

    {
        std::ofstream os(archiveFilename, std::ios::binary);
        cereal::BinaryOutputArchive archive(os);
        archive(rlm);
    }

    {
        auto fromFile = createModel<ResonantLevel>();
        {
            std::ifstream is(archiveFilename, std::ios::binary);
            cereal::BinaryInputArchive archive(is);
            archive(fromFile);
        }

        EXPECT_EQ(dynamic_cast<ResonantLevel&>(*rlm), dynamic_cast<ResonantLevel&>(*fromFile));
    }
}
