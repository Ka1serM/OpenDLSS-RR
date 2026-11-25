#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>

#include "OpenDlssRr/LibraryRange.h"
#include "OpenDlssRr/ShaderCode.h"
#include "OpenDlssRr/Vulkan/Error.h"

namespace OpenDlssRr {

using Error = Vulkan::Error;

//! The functions below load Vulkan from getInstanceProcAddr, or if it is null from the loader; the library never links it.

//! The device extensions that the host must enable, in addition to Vulkan 1.3.
[[nodiscard]] std::span<const char* const> getDeviceExtensions();
//! The extensions of getDeviceExtensions() that the physical device has; a host that must not fail to open its device enables only these.
[[nodiscard]] std::expected<std::vector<const char*>, Error> getSupportedDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice,
                                                                                          PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr);
//! Fails with the first missing device feature that the kernels need; the host must enable all of them (docs/runtime.md).
[[nodiscard]] std::expected<void, Error> checkDevice(VkInstance instance, VkPhysicalDevice physicalDevice,
                                                     PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr);

//! The folder where the user puts the DLSS Ray Reconstruction library, under any name; findLibrary() creates it when it finds nothing.
[[nodiscard]] std::filesystem::path getLibraryFolder();
//! The DLSS Ray Reconstruction library that open() reads with Settings::library (see there), or why there is none.
[[nodiscard]] std::expected<std::filesystem::path, Error> findLibrary(const std::filesystem::path& library = {});
//! Whether the bytes are the library or its libraryRange (LibraryRange.h), by size and SHA-256; a host checks a download before it keeps it.
[[nodiscard]] std::expected<void, Error> checkLibrary(std::span<const std::byte> bytes);

//! An image of the host in VK_IMAGE_LAYOUT_GENERAL with one mip level and one layer. Inputs need sampled use, the output storage use.
struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

//! The images of one evaluation: render size inputs, an output size output in R16G16B16A16_SFLOAT.
struct Textures {
    Texture color;
    Texture diffuseAlbedo;
    Texture specularAlbedo;
    Texture normals;
    Texture roughness;
    Texture depth;
    Texture motionVectors;
    Texture specularHitDistance;
    Texture output;
};

//! The per-frame values of an evaluation. The render subrect, motion vector scale and exposure scales are fixed (docs/runtime.md).
struct Evaluation {
    //! InJitterOffsetX and InJitterOffsetY, in render pixels.
    std::array<float, 2> jitter{};
    bool reset = false;
    //! pInWorldToViewMatrix and pInViewToClipMatrix, row major with row vectors (v * M).
    std::array<float, 16> worldToView{};
    std::array<float, 16> viewToClip{};
};

struct Settings {
    VkInstance instance = VK_NULL_HANDLE;
    //! Optional: the vkGetInstanceProcAddr of the host; the one of the loader if null.
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    //! Optional: a pipeline cache of the host.
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    //! Optional: the path of the DLSS Ray Reconstruction library, under any name; if empty, findLibrary() searches (docs/weights.md).
    std::filesystem::path library;
    //! Optional: the bytes of the library, or of its libraryRange, read in place of `library`; open() does not keep them.
    std::span<const std::byte> libraryData;
    ShaderCode shaders;
    std::uint32_t renderWidth = 0;
    std::uint32_t renderHeight = 0;
    std::uint32_t outputWidth = 0;
    std::uint32_t outputHeight = 0;
};

//! Records into command buffers of the host and never submits (docs/runtime.md).
class RayReconstruction {
public:
    RayReconstruction();
    RayReconstruction(const RayReconstruction&) = delete;
    RayReconstruction& operator=(const RayReconstruction&) = delete;
    RayReconstruction(RayReconstruction&&) noexcept;
    RayReconstruction& operator=(RayReconstruction&&) noexcept;
    //! The GPU must be done with all commands that this object recorded.
    ~RayReconstruction();

    //! Reads the weights and creates the resources; the first record() uploads the weights.
    [[nodiscard]] std::expected<void, Error> open(const Settings& settings);
    [[nodiscard]] std::expected<void, Error> record(VkCommandBuffer commands, const Textures& textures, const Evaluation& evaluation);

    [[nodiscard]] explicit operator bool() const { return static_cast<bool>(state); }

private:
    struct State;
    std::unique_ptr<State> state;
};

}
