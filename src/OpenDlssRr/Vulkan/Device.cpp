#include "OpenDlssRr/Vulkan/Device.h"

#include <algorithm>
#include <format>
#include <utility>
#include <vector>

namespace OpenDlssRr::Vulkan {

std::expected<void, Error> checkFeatures(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features features13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &cooperative};
    VkPhysicalDeviceVulkan12Features features12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &features13};
    VkPhysicalDeviceVulkan11Features features11{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &features12};
    VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features11};
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
    VkPhysicalDeviceSubgroupProperties subgroup{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &subgroup};
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties);
    const std::array<std::pair<const char*, VkBool32>, 11> needed{{
        {"shaderInt64", features.features.shaderInt64},
        {"shaderInt16", features.features.shaderInt16},
        {"shaderStorageImageExtendedFormats", features.features.shaderStorageImageExtendedFormats},
        {"shaderFloat16", features12.shaderFloat16},
        {"storageBuffer16BitAccess", features11.storageBuffer16BitAccess},
        {"uniformAndStorageBuffer16BitAccess", features11.uniformAndStorageBuffer16BitAccess},
        {"shaderBufferInt64Atomics", features12.shaderBufferInt64Atomics},
        {"vulkanMemoryModel", features12.vulkanMemoryModel},
        {"vulkanMemoryModelDeviceScope", features12.vulkanMemoryModelDeviceScope},
        {"cooperativeMatrix", cooperative.cooperativeMatrix},
        {"subgroupSizeControl", subgroup.subgroupSize == 32 || features13.subgroupSizeControl},
    }};
    for (const auto& [feature, supported] : needed)
        if (!supported)
            return fail(std::format("the device has no {}", feature));
    if (!vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        return fail("the Vulkan instance has no vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    std::uint32_t count = 0;
    if (auto result =
            check(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &count, nullptr), "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        !result)
        return result;
    std::vector<VkCooperativeMatrixPropertiesKHR> shapes(count, {.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    if (auto result = check(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &count, shapes.data()),
                            "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        !result)
        return result;
    const auto supports = [&shapes](const VkComponentTypeKHR result) {
        return std::ranges::any_of(shapes, [result](const VkCooperativeMatrixPropertiesKHR& shape) {
            return shape.MSize == 16 && shape.NSize == 16 && shape.KSize == 16 && shape.AType == VK_COMPONENT_TYPE_FLOAT16_KHR
                   && shape.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && shape.CType == result && shape.ResultType == result
                   && shape.scope == VK_SCOPE_SUBGROUP_KHR;
        });
    };
    if (!supports(VK_COMPONENT_TYPE_FLOAT16_KHR) || !supports(VK_COMPONENT_TYPE_FLOAT32_KHR))
        return fail("the device has no 16x16x16 FP16 cooperative matrices with FP16 and FP32 accumulators at subgroup scope");
    return {};
}

void Device::borrow(VkPhysicalDevice physical, VkDevice logical, VkPipelineCache pipelineCache) {
    physicalDevice = physical;
    device = logical;
    cache = pipelineCache;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memory);
}

std::expected<std::uint32_t, Error> Device::memoryType(const std::uint32_t bits, const VkMemoryPropertyFlags flags) const {
    for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index)
        if ((bits & (1u << index)) && (memory.memoryTypes[index].propertyFlags & flags) == flags)
            return index;
    return fail("no matching memory type");
}

}
