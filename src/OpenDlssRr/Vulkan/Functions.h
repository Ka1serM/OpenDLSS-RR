#pragma once

#include <expected>

#include <vulkan/vulkan.h>

#include "OpenDlssRr/Vulkan/Error.h"

//! Built with VK_NO_PROTOTYPES: the commands come from the vkGetInstanceProcAddr of the host, so it works beside volk or Vulkan-Hpp.
#define OPENDLSS_RR_VULKAN_INSTANCE_FUNCTIONS(X)                                                                                                               \
    X(vkAllocateMemory)                                                                                                                                        \
    X(vkBindBufferMemory)                                                                                                                                      \
    X(vkBindImageMemory)                                                                                                                                       \
    X(vkCmdBindPipeline)                                                                                                                                       \
    X(vkCmdClearColorImage)                                                                                                                                    \
    X(vkCmdCopyBuffer)                                                                                                                                         \
    X(vkCmdCopyBufferToImage)                                                                                                                                  \
    X(vkCmdCopyImageToBuffer)                                                                                                                                  \
    X(vkCmdDispatch)                                                                                                                                           \
    X(vkCmdFillBuffer)                                                                                                                                         \
    X(vkCmdPipelineBarrier)                                                                                                                                    \
    X(vkCmdPushConstants)                                                                                                                                      \
    X(vkCmdUpdateBuffer)                                                                                                                                       \
    X(vkCreateBuffer)                                                                                                                                          \
    X(vkCreateComputePipelines)                                                                                                                                \
    X(vkCreateDescriptorSetLayout)                                                                                                                             \
    X(vkCreateImage)                                                                                                                                           \
    X(vkCreateImageView)                                                                                                                                       \
    X(vkCreatePipelineLayout)                                                                                                                                  \
    X(vkCreateSampler)                                                                                                                                         \
    X(vkCreateShaderModule)                                                                                                                                    \
    X(vkDestroyBuffer)                                                                                                                                         \
    X(vkDestroyDescriptorSetLayout)                                                                                                                            \
    X(vkDestroyImage)                                                                                                                                          \
    X(vkDestroyImageView)                                                                                                                                      \
    X(vkDestroyPipeline)                                                                                                                                       \
    X(vkDestroyPipelineLayout)                                                                                                                                 \
    X(vkDestroySampler)                                                                                                                                        \
    X(vkDestroyShaderModule)                                                                                                                                   \
    X(vkEnumerateDeviceExtensionProperties)                                                                                                                    \
    X(vkFreeMemory)                                                                                                                                            \
    X(vkGetBufferMemoryRequirements)                                                                                                                           \
    X(vkGetDeviceProcAddr)                                                                                                                                     \
    X(vkGetImageMemoryRequirements)                                                                                                                            \
    X(vkGetPhysicalDeviceFeatures2)                                                                                                                            \
    X(vkGetPhysicalDeviceMemoryProperties)                                                                                                                     \
    X(vkGetPhysicalDeviceProperties2)                                                                                                                          \
    X(vkMapMemory)

namespace OpenDlssRr {

//! Names in this namespace hide the prototypes of vulkan.h, so the library calls these pointers.
inline PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
#define OPENDLSS_RR_DECLARE_VULKAN_FUNCTION(name) inline PFN_##name name = nullptr;
OPENDLSS_RR_VULKAN_INSTANCE_FUNCTIONS(OPENDLSS_RR_DECLARE_VULKAN_FUNCTION)
#undef OPENDLSS_RR_DECLARE_VULKAN_FUNCTION
//! Of VK_KHR_cooperative_matrix: null when the loader does not know it, so checkFeatures() can name it.
inline PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = nullptr;

namespace Vulkan {

//! Without getInstanceProcAddr it opens the loader (libvulkan.so.1, vulkan-1.dll).
[[nodiscard]] std::expected<void, Error> loadFunctions(VkInstance instance, PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr);

}

}
