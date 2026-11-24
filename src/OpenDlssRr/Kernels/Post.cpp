#include "OpenDlssRr/Kernels/Post.h"

#include <array>

namespace OpenDlssRr::Kernels {

std::expected<void, Vulkan::Error> Post::open(const Vulkan::Device& device, const std::span<const std::byte> spirv) {
    if (auto result = linear.open(device, VK_FILTER_LINEAR); !result)
        return result;
    constexpr VkDescriptorType buffer = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    constexpr VkDescriptorType sampled = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    constexpr VkDescriptorType filtered = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    constexpr VkDescriptorType storage = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    constexpr std::array descriptors{buffer, buffer, sampled, filtered, sampled, filtered, sampled, sampled, filtered, storage, storage};
    return pipeline.open(device, spirv, "post", descriptors, sizeof(Constants));
}

void Post::record(VkCommandBuffer commands, const Resources& resources, const Constants& constants) const {
    const auto buffer = [](const Vulkan::Buffer& source) { return Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .buffer = source.handle()}; };
    const auto sampled = [](const Vulkan::Image& image) { return Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .view = image.view()}; };
    const auto filtered = [this](const Vulkan::Image& image) {
        return Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .view = image.view(), .sampler = linear.handle()};
    };
    const auto storage = [](const Vulkan::Image& image) { return Vulkan::Binding{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .view = image.rawView()}; };
    const std::array bindings{buffer(resources.data),      buffer(resources.exposure),  sampled(resources.guide),   filtered(resources.candidateWeights),
                              sampled(resources.shape),    filtered(resources.history), sampled(resources.motionA), sampled(resources.motionC),
                              filtered(resources.motionB), storage(resources.target),   storage(resources.mip)};
    const std::uint32_t width = static_cast<std::uint32_t>(constants.end[0] - constants.origin[0]);
    const std::uint32_t height = static_cast<std::uint32_t>(constants.end[1] - constants.origin[1]);
    pipeline.dispatch(commands, bindings, &constants, (width + 15) / 16, (height + 31) / 32, 1);
}

}
