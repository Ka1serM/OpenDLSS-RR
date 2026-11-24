#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "OpenDlssRr/Vulkan/ComputePipeline.h"
#include "OpenDlssRr/Vulkan/Resources.h"

namespace OpenDlssRr::Kernels {

//! The network output to the 3x output size (anisotropic Gaussian filter), blended with the reprojected history.
class Post {
public:
    //! The parameters, in the order of the push constants.
    struct Constants {
        //! Output pixel center to render position: fma(x + 0.5, scale, offset).
        float scale[2];
        float offset[2];
        std::int32_t outputSize[2];
        float historyInverseSize[2];
        std::int32_t inputSize[2];
        std::int32_t origin[2];
        std::int32_t end[2];
        //! Width and height of the two FP16 output arrays, which start at the origin.
        std::int32_t pitch[2];
        float inputInverseSize[2];
        float blendScale;
        //! Texels per row of the source array.
        std::int32_t sourcePitch;
        //! Byte offsets in the data buffer: source texels (4 FP16 each), red-green half2 array, blue half array.
        std::uint32_t sourceOffset;
        std::uint32_t colorOffset;
        std::uint32_t blueOffset;
        //! Nonzero: ignore the history.
        std::uint32_t reset;
    };

    //! The images. Target and mip need the raw views R32G32_UINT and R32_UINT.
    struct Resources {
        const Vulkan::Buffer& data;
        const Vulkan::Buffer& exposure;
        const Vulkan::Image& guide;
        const Vulkan::Image& candidateWeights;
        const Vulkan::Image& shape;
        const Vulkan::Image& history;
        const Vulkan::Image& motionA;
        const Vulkan::Image& motionC;
        const Vulkan::Image& motionB;
        const Vulkan::Image& target;
        const Vulkan::Image& mip;
    };

    [[nodiscard]] std::expected<void, Vulkan::Error> open(const Vulkan::Device& device, std::span<const std::byte> spirv);
    void record(VkCommandBuffer commands, const Resources& resources, const Constants& constants) const;

private:
    Vulkan::ComputePipeline pipeline;
    Vulkan::Sampler linear;
};

}
