#include "OpenDlssRr/Kernels/Downsample.h"

#include <array>

namespace OpenDlssRr::Kernels {

std::expected<void, Vulkan::Error> Downsample::open(const Vulkan::Device& device, const std::span<const std::byte> spirv) {
    constexpr std::array descriptors{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    return pipeline.open(device, spirv, "downsample", descriptors, sizeof(Constants));
}

void Downsample::record(VkCommandBuffer commands, const Vulkan::Buffer& data, const Vulkan::Image& output, const Vulkan::Buffer& exposure,
                        const Constants& constants) const {
    const std::array bindings{Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer = data.handle()},
                              Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .view = output.view()},
                              Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer = exposure.handle()}};
    const std::uint32_t width = static_cast<std::uint32_t>(constants.end[0] - constants.origin[0]);
    const std::uint32_t height = static_cast<std::uint32_t>(constants.end[1] - constants.origin[1]);
    pipeline.dispatch(commands, bindings, &constants, (width + 15) / 16, (height + 15) / 16, 1);
}

}
