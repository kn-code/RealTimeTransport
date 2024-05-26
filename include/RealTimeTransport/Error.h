//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

///
/// \file   Error.h
///
/// \brief  Contains exception/error classes.
///

#ifndef REAL_TIME_TRANSPORT_ERROR_H
#define REAL_TIME_TRANSPORT_ERROR_H

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

#include "RealTimeTransport_export.h"

namespace RealTimeTransport
{

///
/// @brief This class represents a generic error.
///
/// This class represents a generic error.
///
class REALTIMETRANSPORT_EXPORT Error : public std::exception
{
  public:
    /// @brief Default constructor.
    Error() noexcept;

    /// @brief Constructs an error with a given message.
    Error(const std::string& message, std::source_location location = std::source_location::current());

    /// @brief Move constructor.
    Error(Error&& other) noexcept;

    virtual ~Error() noexcept;

    /// @brief Move assignment operator.
    Error& operator=(Error&& other) noexcept;

    /// @brief Returns the error message.
    const char* what() const noexcept override;

  private:
    std::string _what;
};

///
/// @brief Class representing the result of a computation that didn't meet a required accuracy goal.
///
/// This class represents the result of a computation that didn't meet a required accuracy goal.
/// The class contains an error message and the preliminary result of the computation.
///
/// @tparam T   Result type of the computation.
///
template <typename T>
class REALTIMETRANSPORT_EXPORT AccuracyError : public Error
{
  public:
    /// @brief Constructs an error with a message and a preliminary value.
    AccuracyError(
        const std::string& message,
        T&& value,
        std::source_location location = std::source_location::current())
        : Error(message, location), _value(std::move(value))
    {
    }

    /// @brief Constructs an error from an \a error instance and a preliminary value.
    AccuracyError(Error&& error, T&& value) : Error(std::move(error)), _value(std::move(value))
    {
    }

    /// @brief Return the preliminary result of the computation.
    T& value() noexcept
    {
        return _value;
    }

    /// @brief Return the preliminary result of the computation.
    const T& value() const noexcept
    {
        return _value;
    }

  private:
    T _value;
};

} // namespace RealTimeTransport

#endif // REAL_TIME_TRANSPORT_ERROR_H
