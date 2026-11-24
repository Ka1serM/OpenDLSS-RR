#pragma once

#include <array>
#include <cstdint>
#include <expected>

#include "OpenDlssRr/Vulkan/Error.h"
#include "OpenDlssRr/Vulkan/Functions.h"

namespace OpenDlssRr::Vulkan {

//! The Vulkan device of the host renderer, with its memory types and pipeline cache.
class Device {
public:
    //! Uses the device of the host. The host must enable deviceExtensions and the features of checkFeatures().
    void borrow(VkPhysicalDevice physical, VkDevice device, VkPipelineCache pipelineCache);

    [[nodiscard]] std::expected<std::uint32_t, Error> memoryType(std::uint32_t bits, VkMemoryPropertyFlags flags) const;

    [[nodiscard]] VkPipelineCache pipelineCache() const { return cache; }
    [[nodiscard]] VkPhysicalDevice physical() const { return physicalDevice; }
    [[nodiscard]] VkDevice handle() const { return device; }

private:
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPipelineCache cache = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
};

//! The device extensions that the runtime needs, in addition to Vulkan 1.3.
inline constexpr std::array<const char*, 2> deviceExtensions{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
//! Fails with the first missing device feature that the runtime needs (docs/runtime.md).
[[nodiscard]] std::expected<void, Error> checkFeatures(VkPhysicalDevice physicalDevice);

}
