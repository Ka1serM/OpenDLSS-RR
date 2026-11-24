#pragma once

#include "OpenDlssRr/Vulkan/Resources.h"

namespace OpenDlssRr::Kernels {

//! Sets whole 4-byte words of a buffer to zero with a transfer fill.
inline void recordClearRange(VkCommandBuffer commands, const Vulkan::Buffer& buffer, const VkDeviceSize offset, const VkDeviceSize bytes) {
    vkCmdFillBuffer(commands, buffer.handle(), offset, bytes, 0);
}

}
