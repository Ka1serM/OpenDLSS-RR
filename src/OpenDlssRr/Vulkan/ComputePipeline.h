#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "OpenDlssRr/Vulkan/Device.h"

namespace OpenDlssRr::Vulkan {

//! One descriptor of a push descriptor set: a sampled image with its sampler, a storage image, or a storage or uniform buffer.
struct Binding {
    VkDescriptorType type;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
};

//! A compute pipeline from SPIR-V, with one push descriptor set and push constants.
class ComputePipeline {
public:
    ComputePipeline() = default;
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ~ComputePipeline();

    [[nodiscard]] std::expected<void, Error> open(const Device& device, std::span<const std::byte> spirv, const char* entryPoint,
                                                  std::span<const VkDescriptorType> descriptors, std::uint32_t pushConstantBytes,
                                                  std::span<const std::uint32_t> specialization = {}, std::uint32_t subgroupSize = 0);
    void dispatch(VkCommandBuffer commands, std::span<const Binding> bindings, const void* pushConstants, std::uint32_t x, std::uint32_t y,
                  std::uint32_t z) const;

private:
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::uint32_t constantBytes = 0;
    std::vector<VkDescriptorType> types;
    PFN_vkCmdPushDescriptorSetKHR pushDescriptorSet = nullptr;
};

}
