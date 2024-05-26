//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#include "RealTimeTransport/Error.h"

#include <sstream>

namespace RealTimeTransport
{

Error::Error() noexcept
{
}

Error::Error(const std::string& message, std::source_location location)
{
    std::stringstream ss;
    ss << location.file_name() << ':' << location.line() << ": error: " << message << " [function `"
       << location.function_name() << "`]";
    _what = ss.str();
}

Error::Error(Error&& other) noexcept : _what(std::move(other._what))
{
}

Error::~Error() noexcept
{
}

Error& Error::operator=(Error&& other) noexcept
{
    _what = std::move(other._what);
    return *this;
}

const char* Error::what() const noexcept
{
    return _what.c_str();
}

} // namespace RealTimeTransport
