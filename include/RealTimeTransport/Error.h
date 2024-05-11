//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#ifndef REAL_TIME_TRANSPORT_ERROR_H
#define REAL_TIME_TRANSPORT_ERROR_H

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

#include "RealTimeTransport_export.h"

namespace RealTimeTransport
{

class REALTIMETRANSPORT_EXPORT Error : public std::exception
{
  public:
    Error(std::string_view message, std::source_location location = std::source_location::current());

    const char* what() const noexcept override;

  private:
    std::string _what;
};

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_ERROR_H
