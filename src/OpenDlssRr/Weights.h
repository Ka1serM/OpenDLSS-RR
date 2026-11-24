#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
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
    //! The library to read: `library`, else OPENDLSS_RR_LIBRARY, else the first supported file in the search folders.
    [[nodiscard]] static std::expected<std::filesystem::path, Vulkan::Error> find(const std::filesystem::path& library);
    //! Reads a supported library: libnvidia-ngx-dlssd.so.310.9.16 of the DLSS SDK or libnvidia-ngx-builtin-dlssd.so.310.9.0 of the driver.
    [[nodiscard]] std::expected<void, Vulkan::Error> open(const std::filesystem::path& library);

    //! Name ("b16.t08": buffer 16, tensor 8) to tensor.
    std::map<std::string, WeightTensor, std::less<>> tensors;
    //! The two factors of the attention bias tables: (M + transpose(M)) * pairScale[0] * pairScale[1].
    std::array<float, 2> pairScale{};
    //! Per-kernel scalars, such as "dec4_0.skip".
    std::map<std::string, float, std::less<>> constants;
};

}
