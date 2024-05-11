//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/Error.h"

#include <sstream>

namespace RealTimeTransport
{

Error::Error(std::string_view message, std::source_location location)
{
    std::stringstream ss;
    ss << location.file_name() << ':' << location.line() << ": error: " << message << "[function `"
       << location.function_name() << "`]";
    _what = ss.str();
}

const char* Error::what() const noexcept
{
    return _what.c_str();
}

} // namespace RealTimeTransport
