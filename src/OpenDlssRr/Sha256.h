#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string>

namespace OpenDlssRr {

//! The SHA-256 digest of the bytes, as 64 lowercase hex digits.
[[nodiscard]] std::string sha256(std::span<const std::byte> bytes);

}
