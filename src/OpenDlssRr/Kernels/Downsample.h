#pragma once

#include <cstdint>

#include "OpenDlssRr/Vulkan/ComputePipeline.h"
#include "OpenDlssRr/Vulkan/Resources.h"

namespace OpenDlssRr::Kernels {

//! The network output at 1.5x size to the output image, in linear light.
class Downsample {
public:
    //! The parameters, in the order of the push constants.
    struct Constants {
        std::int32_t sourceSize[2];
        std::int32_t outputSize[2];
        std::int32_t outputOffset[2];
        std::int32_t origin[2];
        std::int32_t end[2];
        //! Width and height of the stored source region.
        std::int32_t pitch[2];
        std::int32_t regionBegin[2];
        std::int32_t regionEnd[2];
        //! Byte offsets in the data buffer of the red-green half2 array and of the blue half array.
        std::uint32_t colorOffset;
        std::uint32_t blueOffset;
    };

    [[nodiscard]] std::expected<void, Vulkan::Error> open(const Vulkan::Device& device, std::span<const std::byte> spirv);
    //! The output needs storage use. The exposure buffer starts with the FP32 exposure.
    void record(VkCommandBuffer commands, const Vulkan::Buffer& data, const Vulkan::Image& output, const Vulkan::Buffer& exposure,
                const Constants& constants) const;

private:
    Vulkan::ComputePipeline pipeline;
};

}
