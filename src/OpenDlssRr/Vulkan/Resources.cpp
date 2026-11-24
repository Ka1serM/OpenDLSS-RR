#include "OpenDlssRr/Vulkan/Resources.h"

#include <utility>

namespace OpenDlssRr::Vulkan {

Buffer::Buffer(Buffer&& other) noexcept {
    *this = std::move(other);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        release();
        device = std::exchange(other.device, VK_NULL_HANDLE);
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        memory = std::exchange(other.memory, VK_NULL_HANDLE);
        bytes = std::exchange(other.bytes, 0);
        pointer = std::exchange(other.pointer, nullptr);
    }
    return *this;
}

Buffer::~Buffer() {
    release();
}

void Buffer::release() {
    if (buffer)
        vkDestroyBuffer(device, buffer, nullptr);
    if (memory)
        vkFreeMemory(device, memory, nullptr);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}

std::expected<void, Error> Buffer::open(const Device& owner, const VkDeviceSize size, const bool hostVisible) {
    release();
    device = owner.handle();
    bytes = size;
    const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .size = size,
                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                           | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    if (auto result = check(vkCreateBuffer(device, &info, nullptr, &buffer), "vkCreateBuffer"); !result)
        return result;
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    //! Host memory is cached for reads where the device has such a type.
    constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags wanted = hostVisible ? host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT : VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    std::expected<std::uint32_t, Error> type = owner.memoryType(requirements.memoryTypeBits, wanted);
    if (!type && hostVisible)
        type = owner.memoryType(requirements.memoryTypeBits, host);
    if (!type)
        return std::unexpected(type.error());
    const VkMemoryAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = requirements.size, .memoryTypeIndex = *type};
    if (auto result = check(vkAllocateMemory(device, &allocate, nullptr, &memory), "vkAllocateMemory"); !result)
        return result;
    if (auto result = check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory"); !result)
        return result;
    if (hostVisible)
        return check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &pointer), "vkMapMemory");
    return {};
}

Image::Image(Image&& other) noexcept {
    *this = std::move(other);
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        release();
        device = std::exchange(other.device, VK_NULL_HANDLE);
        image = std::exchange(other.image, VK_NULL_HANDLE);
        imageView = std::exchange(other.imageView, VK_NULL_HANDLE);
        rawImageView = std::exchange(other.rawImageView, VK_NULL_HANDLE);
        memory = std::exchange(other.memory, VK_NULL_HANDLE);
        imageFormat = other.imageFormat;
        extentWidth = other.extentWidth;
        extentHeight = other.extentHeight;
    }
    return *this;
}

Image::~Image() {
    release();
}

void Image::release() {
    if (!device) {
        image = VK_NULL_HANDLE;
        imageView = VK_NULL_HANDLE;
        return;
    }
    if (rawImageView)
        vkDestroyImageView(device, rawImageView, nullptr);
    if (imageView)
        vkDestroyImageView(device, imageView, nullptr);
    if (image)
        vkDestroyImage(device, image, nullptr);
    if (memory)
        vkFreeMemory(device, memory, nullptr);
    rawImageView = VK_NULL_HANDLE;
    imageView = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}

std::expected<void, Error> Image::open(const Device& owner, const VkFormat format, const std::uint32_t width, const std::uint32_t height,
                                       const VkFormat rawFormat) {
    release();
    device = owner.handle();
    imageFormat = format;
    extentWidth = width;
    extentHeight = height;
    const VkImageCreateInfo info{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                 .flags = rawFormat != VK_FORMAT_UNDEFINED ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0u,
                                 .imageType = VK_IMAGE_TYPE_2D,
                                 .format = format,
                                 .extent = {width, height, 1},
                                 .mipLevels = 1,
                                 .arrayLayers = 1,
                                 .samples = VK_SAMPLE_COUNT_1_BIT,
                                 .tiling = VK_IMAGE_TILING_OPTIMAL,
                                 .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                          | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                 .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    if (auto result = check(vkCreateImage(device, &info, nullptr, &image), "vkCreateImage"); !result)
        return result;
    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(device, image, &requirements);
    const std::expected<std::uint32_t, Error> type = owner.memoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!type)
        return std::unexpected(type.error());
    const VkMemoryAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = requirements.size, .memoryTypeIndex = *type};
    if (auto result = check(vkAllocateMemory(device, &allocate, nullptr, &memory), "vkAllocateMemory"); !result)
        return result;
    if (auto result = check(vkBindImageMemory(device, image, memory, 0), "vkBindImageMemory"); !result)
        return result;
    const VkImageViewCreateInfo viewInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                         .image = image,
                                         .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                         .format = format,
                                         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (auto result = check(vkCreateImageView(device, &viewInfo, nullptr, &imageView), "vkCreateImageView"); !result || rawFormat == VK_FORMAT_UNDEFINED)
        return result;
    VkImageViewCreateInfo rawInfo = viewInfo;
    rawInfo.format = rawFormat;
    return check(vkCreateImageView(device, &rawInfo, nullptr, &rawImageView), "vkCreateImageView");
}

void Image::borrow(VkImage handle, VkImageView view, const VkFormat format, const std::uint32_t width, const std::uint32_t height) {
    release();
    //! Without a device, release() only forgets the handles.
    device = VK_NULL_HANDLE;
    rawImageView = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    image = handle;
    imageView = view;
    imageFormat = format;
    extentWidth = width;
    extentHeight = height;
}

void Image::initialize(VkCommandBuffer commands) const {
    const VkImageMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                       .srcAccessMask = 0,
                                       .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                       .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                       .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .image = image,
                                       .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    const VkClearColorValue zero{};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(commands, image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    fullBarrier(commands);
}

void Image::upload(VkCommandBuffer commands, const Buffer& staging, const VkDeviceSize offset) const {
    const VkBufferImageCopy region{
        .bufferOffset = offset, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {extentWidth, extentHeight, 1}};
    vkCmdCopyBufferToImage(commands, staging.handle(), image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

void Image::download(VkCommandBuffer commands, const Buffer& staging, const VkDeviceSize offset) const {
    const VkBufferImageCopy region{
        .bufferOffset = offset, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {extentWidth, extentHeight, 1}};
    vkCmdCopyImageToBuffer(commands, image, VK_IMAGE_LAYOUT_GENERAL, staging.handle(), 1, &region);
}

Sampler::Sampler(Sampler&& other) noexcept {
    *this = std::move(other);
}

Sampler& Sampler::operator=(Sampler&& other) noexcept {
    if (this != &other) {
        release();
        device = std::exchange(other.device, VK_NULL_HANDLE);
        sampler = std::exchange(other.sampler, VK_NULL_HANDLE);
    }
    return *this;
}

Sampler::~Sampler() {
    release();
}

void Sampler::release() {
    if (sampler)
        vkDestroySampler(device, sampler, nullptr);
    sampler = VK_NULL_HANDLE;
}

std::expected<void, Error> Sampler::open(const Device& owner, const VkFilter filter) {
    release();
    device = owner.handle();
    const VkSamplerCreateInfo info{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                   .magFilter = filter,
                                   .minFilter = filter,
                                   .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                   .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                   .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                   .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                   .maxAnisotropy = 1.0f,
                                   .minLod = 0.0f,
                                   .maxLod = 0.0f};
    return check(vkCreateSampler(device, &info, nullptr, &sampler), "vkCreateSampler");
}

std::uint32_t texelBytes(const VkFormat format) {
    switch (format) {
    case VK_FORMAT_R16_SFLOAT:
        return 2;
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_UINT:
        return 8;
    default:
        return 0;
    }
}

void fullBarrier(VkCommandBuffer commands) {
    const VkMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                                  .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

}
