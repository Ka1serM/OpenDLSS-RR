#pragma once

#include <cstdint>
#include <expected>

#include "OpenDlssRr/Vulkan/Device.h"

namespace OpenDlssRr::Vulkan {

//! A storage or uniform buffer, optionally mapped for host access.
class Buffer {
public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    ~Buffer();

    [[nodiscard]] std::expected<void, Error> open(const Device& device, VkDeviceSize size, bool hostVisible);

    [[nodiscard]] VkBuffer handle() const { return buffer; }
    [[nodiscard]] VkDeviceSize size() const { return bytes; }
    [[nodiscard]] void* mapped() const { return pointer; }

private:
    void release();

    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize bytes = 0;
    void* pointer = nullptr;
};

//! A 2D image with one mip level, used as sampled and storage image, in VK_IMAGE_LAYOUT_GENERAL after initialize().
class Image {
public:
    Image() = default;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;
    ~Image();

    //! A raw format other than VK_FORMAT_UNDEFINED adds a second view of that format (R32_UINT for exact stores of packed texels).
    [[nodiscard]] std::expected<void, Error> open(const Device& device, VkFormat format, std::uint32_t width, std::uint32_t height,
                                                  VkFormat rawFormat = VK_FORMAT_UNDEFINED);
    //! Uses an image of the host, which keeps it; the view must have the format.
    void borrow(VkImage handle, VkImageView view, VkFormat format, std::uint32_t width, std::uint32_t height);
    //! Moves the image to VK_IMAGE_LAYOUT_GENERAL and clears it to zero.
    void initialize(VkCommandBuffer commands) const;
    void upload(VkCommandBuffer commands, const Buffer& staging, VkDeviceSize offset) const;
    void download(VkCommandBuffer commands, const Buffer& staging, VkDeviceSize offset) const;

    [[nodiscard]] VkImage handle() const { return image; }
    [[nodiscard]] VkImageView view() const { return imageView; }
    [[nodiscard]] VkImageView rawView() const { return rawImageView; }
    [[nodiscard]] VkFormat format() const { return imageFormat; }
    [[nodiscard]] std::uint32_t width() const { return extentWidth; }
    [[nodiscard]] std::uint32_t height() const { return extentHeight; }

private:
    void release();

    VkDevice device = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkImageView rawImageView = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat imageFormat = VK_FORMAT_UNDEFINED;
    std::uint32_t extentWidth = 0;
    std::uint32_t extentHeight = 0;
};

//! A sampler with clamp to edge, normalized coordinates and LOD 0.
class Sampler {
public:
    Sampler() = default;
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&& other) noexcept;
    Sampler& operator=(Sampler&& other) noexcept;
    ~Sampler();

    [[nodiscard]] std::expected<void, Error> open(const Device& device, VkFilter filter);

    [[nodiscard]] VkSampler handle() const { return sampler; }

private:
    void release();

    VkDevice device = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

//! Bytes of one texel of the formats that the port uses.
[[nodiscard]] std::uint32_t texelBytes(VkFormat format);

//! A full barrier between compute and transfer work, for a correct (not a fast) schedule.
void fullBarrier(VkCommandBuffer commands);

}
