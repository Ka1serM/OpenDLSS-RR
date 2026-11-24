#pragma once

#include "OpenDlssRr/Vulkan/ComputePipeline.h"
#include "OpenDlssRr/Vulkan/Resources.h"

namespace OpenDlssRr::Kernels {

//! The history log image (A2B10G10R10) at half size, averaged in linear light.
class PooledHistory {
public:
    [[nodiscard]] std::expected<void, Vulkan::Error> open(const Vulkan::Device& device, std::span<const std::byte> spirv);
    //! The target needs a VK_FORMAT_R32_UINT raw view.
    void record(VkCommandBuffer commands, const Vulkan::Image& source, const Vulkan::Image& target) const;

private:
    Vulkan::ComputePipeline pipeline;
};

}
