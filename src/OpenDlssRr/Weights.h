#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "OpenDlssRr/Vulkan/Error.h"

namespace OpenDlssRr {

//! The value of FP16 bits (std::float16_t is not in every standard library).
[[nodiscard]] float halfToFloat(std::uint16_t half);

//! A row-major FP16 tensor in its source layout; "lut" tensors are already rounded to E4M3 values.
struct WeightTensor {
    std::vector<std::uint32_t> shape;
    std::vector<std::uint16_t> values;
};

class Weights {
public:
    //! The OpenDLSS-RR folder of the user data, where the user puts the library: $XDG_DATA_HOME or ~/.local/share, %LOCALAPPDATA%.
    [[nodiscard]] static std::filesystem::path folder();
    //! `library` if it is not empty, else OPENDLSS_RR_LIBRARY, else the file in folder().
    [[nodiscard]] static std::expected<std::filesystem::path, Vulkan::Error> find(const std::filesystem::path& library);
    //! Reads libnvidia-ngx-dlssd.so.310.9.16 of the DLSS SDK, under any name.
    [[nodiscard]] std::expected<void, Vulkan::Error> open(const std::filesystem::path& library);
    //! Reads the bytes of the library from memory.
    [[nodiscard]] std::expected<void, Vulkan::Error> open(std::span<const std::byte> bytes);

    //! Name ("b16.t08": buffer 16, tensor 8) to tensor.
    std::map<std::string, WeightTensor, std::less<>> tensors;
    //! The two factors of the attention bias tables: (M + transpose(M)) * pairScale[0] * pairScale[1].
    std::array<float, 2> pairScale{};
    //! Per-kernel scalars, such as "dec4_0.skip".
    std::map<std::string, float, std::less<>> constants;
};

}
