#pragma once

#include "OpenDlssRr/Vulkan/ComputePipeline.h"
#include "OpenDlssRr/Vulkan/Resources.h"

namespace OpenDlssRr::Kernels {

//! The exposure from the mean luminance of the colour input; the accumulators and exposure images alternate each frame.
class HistogramBasic {
public:
    struct Resources {
        const Vulkan::Image& color;
        const Vulkan::Image& exposure;
        const Vulkan::Image& previousExposure;
        const Vulkan::Buffer& current;
        const Vulkan::Buffer& next;
        const Vulkan::Buffer& result;
    };

    [[nodiscard]] std::expected<void, Vulkan::Error> open(const Vulkan::Device& device, std::span<const std::byte> spirv);
    //! reset is true for the first frame: then the exposure change is 0.
    void record(VkCommandBuffer commands, const Resources& resources, bool reset) const;

private:
    Vulkan::ComputePipeline pipeline;
};

}
