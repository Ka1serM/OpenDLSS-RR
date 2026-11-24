#include "OpenDlssRr/Kernels/PooledHistory.h"

#include <array>
#include <cstdint>

namespace OpenDlssRr::Kernels {

namespace {

struct Constants {
    std::int32_t sourceSize[2];
    std::int32_t targetSize[2];
};

}

std::expected<void, Vulkan::Error> PooledHistory::open(const Vulkan::Device& device, const std::span<const std::byte> spirv) {
    constexpr std::array descriptors{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
    return pipeline.open(device, spirv, "pooledHistory", descriptors, sizeof(Constants));
}

void PooledHistory::record(VkCommandBuffer commands, const Vulkan::Image& source, const Vulkan::Image& target) const {
    const Constants constants{{static_cast<std::int32_t>(source.width()), static_cast<std::int32_t>(source.height())},
                              {static_cast<std::int32_t>(target.width()), static_cast<std::int32_t>(target.height())}};
    const std::array bindings{Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .view = source.view()},
                              Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .view = target.rawView()}};
    pipeline.dispatch(commands, bindings, &constants, (target.width() + 15) / 16, (target.height() + 15) / 16, 1);
}

}
