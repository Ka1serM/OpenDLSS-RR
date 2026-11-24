#include "OpenDlssRr/Kernels/HistogramBasic.h"

#include <array>
#include <cstdint>

namespace OpenDlssRr::Kernels {

namespace {

struct Constants {
    std::int32_t offset[2];
    std::uint32_t last[2];
    std::uint32_t blockCount;
    float meanScale;
    std::uint32_t reset;
};

}

std::expected<void, Vulkan::Error> HistogramBasic::open(const Vulkan::Device& device, const std::span<const std::byte> spirv) {
    constexpr std::array descriptors{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    return pipeline.open(device, spirv, "histogramBasic", descriptors, sizeof(Constants));
}

void HistogramBasic::record(VkCommandBuffer commands, const Resources& resources, const bool reset) const {
    const std::uint32_t width = resources.color.width();
    const std::uint32_t height = resources.color.height();
    const std::uint32_t groupsX = (width + 15) / 16;
    const std::uint32_t groupsY = (height + 15) / 16;
    //! It scales the fixed-point sum (2^32 / 10^4 per unit) back to a mean in FP32.
    const float meanScale = static_cast<float>(1.0 / (429496.71875 * width * height));
    const Constants constants{{0, 0}, {width - 1, height - 1}, groupsX * groupsY, meanScale, reset ? 1u : 0u};
    const std::array bindings{Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .view = resources.color.view()},
                              Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .view = resources.exposure.view()},
                              Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .view = resources.previousExposure.view()},
                              Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer = resources.current.handle()},
                              Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer = resources.next.handle()},
                              Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer = resources.result.handle()}};
    pipeline.dispatch(commands, bindings, &constants, groupsX, groupsY, 1);
}

}
