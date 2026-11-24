#include "OpenDlssRr/RayReconstruction.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <string_view>
#include <tuple>

#include "OpenDlssRr/Camera.h"
#include "OpenDlssRr/Kernels/ClearRange.h"
#include "OpenDlssRr/Kernels/Downsample.h"
#include "OpenDlssRr/Kernels/HistogramBasic.h"
#include "OpenDlssRr/Kernels/PooledHistory.h"
#include "OpenDlssRr/Kernels/Post.h"
#include "OpenDlssRr/Network.h"
#include "OpenDlssRr/Vulkan/Device.h"
#include "OpenDlssRr/Vulkan/Resources.h"
#include "OpenDlssRr/Weights.h"

namespace OpenDlssRr {

namespace {

using Vulkan::fail;
using Size = std::array<std::uint32_t, 2>;

Size halve(const Size size) {
    return {(size[0] + 1) / 2, (size[1] + 1) / 2};
}

std::array<float, 2> divide(const Size numerator, const Size denominator) {
    return {static_cast<float>(numerator[0]) / static_cast<float>(denominator[0]), static_cast<float>(numerator[1]) / static_cast<float>(denominator[1])};
}

std::array<float, 2> inverse(const Size size) {
    return {1.0f / static_cast<float>(size[0]), 1.0f / static_cast<float>(size[1])};
}

void copy(float (&target)[2], const std::array<float, 2> source) {
    target[0] = source[0];
    target[1] = source[1];
}

void copy(std::int32_t (&target)[2], const Size source) {
    target[0] = static_cast<std::int32_t>(source[0]);
    target[1] = static_cast<std::int32_t>(source[1]);
}

//! A 4x4 matrix of the camera block (Camera.h) from its first word.
std::array<float, 16> matrix(const std::array<std::uint32_t, cameraWordCount>& words, const std::size_t first) {
    std::array<float, 16> values;
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = std::bit_cast<float>(words[first + index]);
    return values;
}

}

//! The state between frames, with the image numbers of docs/network.md. Pairs alternate with the frame parity.
struct RayReconstruction::State {
    Vulkan::Device device;
    Network network;
    Kernels::Post post;
    Kernels::Downsample downsample;
    Kernels::HistogramBasic histogram;
    Kernels::PooledHistory pooledHistory;
    Size renderSize{};
    Size outputSize{};
    //! The post output: 3 · ceil(output / 2) per axis.
    Size postSize{};
    //! Post output (log encoded, images 9 and 10), its mip (11) and the mip of that (12).
    std::array<Vulkan::Image, 2> history;
    Vulkan::Image mip;
    Vulkan::Image pooledMip;
    Vulkan::Image feedback;
    std::array<Vulkan::Image, 2> motion;
    Vulkan::Image centreMotion;
    Vulkan::Image farthestMotion;
    Vulkan::Image guide;
    Vulkan::Image candidateWeights;
    Vulkan::Image shape;
    //! The exposure images (16 and 17), the exposure accumulators (buffers 2 and 3) and the exposure (buffer 4).
    std::array<Vulkan::Image, 2> exposureImages;
    std::array<Vulkan::Buffer, 2> exposureSums;
    Vulkan::Buffer exposure;
    //! The network colour (FP16 RGBA, L0 size), then the post output as a red-green half2 array and a blue half array (pitch size).
    Vulkan::Buffer postData;
    std::array<Vulkan::Image, 9> inputs;
    bool initialized = false;
    std::uint32_t frameIndex = 0;
    std::array<float, 2> previousJitter{};
    CameraMatrices previousCamera;

    [[nodiscard]] std::expected<void, Error> open(const Settings& settings);
    void initialize(VkCommandBuffer commands) const;
};

std::span<const char* const> getDeviceExtensions() {
    return Vulkan::deviceExtensions;
}

std::expected<std::vector<const char*>, Error> getSupportedDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice,
                                                                            PFN_vkGetInstanceProcAddr getInstanceProcAddr) {
    if (auto result = Vulkan::loadFunctions(instance, getInstanceProcAddr); !result)
        return std::unexpected(result.error());
    std::uint32_t count = 0;
    if (auto result = Vulkan::check(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
        !result)
        return std::unexpected(result.error());
    std::vector<VkExtensionProperties> available(count);
    if (auto result =
            Vulkan::check(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, available.data()), "vkEnumerateDeviceExtensionProperties");
        !result)
        return std::unexpected(result.error());
    std::vector<const char*> supported;
    for (const char* name : Vulkan::deviceExtensions)
        if (std::ranges::any_of(available, [name](const VkExtensionProperties& entry) { return std::string_view(entry.extensionName) == name; }))
            supported.push_back(name);
    return supported;
}

std::expected<void, Error> checkDevice(VkInstance instance, VkPhysicalDevice physicalDevice, PFN_vkGetInstanceProcAddr getInstanceProcAddr) {
    if (auto result = Vulkan::loadFunctions(instance, getInstanceProcAddr); !result)
        return result;
    return Vulkan::checkFeatures(physicalDevice);
}

std::filesystem::path getLibraryFolder() {
    return Weights::folder();
}

std::expected<std::filesystem::path, Error> findLibrary(const std::filesystem::path& library) {
    return Weights::find(library);
}

RayReconstruction::RayReconstruction() = default;
RayReconstruction::RayReconstruction(RayReconstruction&&) noexcept = default;
RayReconstruction& RayReconstruction::operator=(RayReconstruction&&) noexcept = default;
RayReconstruction::~RayReconstruction() = default;

std::expected<void, Error> RayReconstruction::State::open(const Settings& settings) {
    if (auto result = Vulkan::loadFunctions(settings.instance, settings.getInstanceProcAddr); !result)
        return result;
    device.borrow(settings.physicalDevice, settings.device, settings.pipelineCache);
    renderSize = {settings.renderWidth, settings.renderHeight};
    outputSize = {settings.outputWidth, settings.outputHeight};
    if (renderSize[0] == 0 || renderSize[1] == 0 || outputSize[0] < renderSize[0] || outputSize[1] < renderSize[1])
        return fail("the render size must not be 0 or larger than the output size");
    postSize = {3 * ((outputSize[0] + 1) / 2), 3 * ((outputSize[1] + 1) / 2)};

    const auto library = Weights::find(settings.library);
    if (!library)
        return std::unexpected(library.error());
    Weights weights;
    if (auto result = weights.open(*library); !result)
        return result;
    if (auto result = network.open(device, weights, settings.shaders, renderSize); !result)
        return result;
    for (const auto& [kernel, result] :
         {std::pair{"Post", post.open(device, settings.shaders.post)}, std::pair{"Downsample", downsample.open(device, settings.shaders.downsample)},
          std::pair{"HistogramBasic", histogram.open(device, settings.shaders.histogramBasic)},
          std::pair{"PooledHistory", pooledHistory.open(device, settings.shaders.pooledHistory)}})
        if (!result)
            return fail(std::format("{}: {}", kernel, result.error().message));

    const auto image = [&](Vulkan::Image& target, const VkFormat format, const Size size, const VkFormat raw = VK_FORMAT_UNDEFINED) {
        return target.open(device, format, size[0], size[1], raw);
    };
    constexpr VkFormat rgba16 = VK_FORMAT_R16G16B16A16_SFLOAT;
    constexpr VkFormat packed = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    const Size mipSize = halve(postSize);
    for (const auto& result :
         {image(history[0], rgba16, postSize, VK_FORMAT_R32G32_UINT), image(history[1], rgba16, postSize, VK_FORMAT_R32G32_UINT),
          image(mip, packed, mipSize, VK_FORMAT_R32_UINT), image(pooledMip, packed, halve(mipSize), VK_FORMAT_R32_UINT),
          image(feedback, VK_FORMAT_R8G8B8A8_SNORM, renderSize), image(motion[0], rgba16, renderSize), image(motion[1], rgba16, renderSize),
          image(centreMotion, VK_FORMAT_R16G16_SFLOAT, renderSize, VK_FORMAT_R32_UINT),
          image(farthestMotion, VK_FORMAT_R16G16_SFLOAT, renderSize, VK_FORMAT_R32_UINT), image(guide, VK_FORMAT_R8G8B8A8_UNORM, renderSize),
          image(candidateWeights, VK_FORMAT_R8G8B8A8_UNORM, renderSize), image(shape, rgba16, renderSize),
          image(exposureImages[0], VK_FORMAT_R16_SFLOAT, {1, 1}), image(exposureImages[1], VK_FORMAT_R16_SFLOAT, {1, 1})})
        if (!result)
            return result;
    const Size level0 = network.size();
    const Size pitch = {(postSize[0] + 1) / 2 * 2, (postSize[1] + 1) / 2 * 2};
    const VkDeviceSize postBytes = VkDeviceSize{8} * level0[0] * level0[1] + VkDeviceSize{6} * pitch[0] * pitch[1];
    for (const auto& result : {exposureSums[0].open(device, 1028, false), exposureSums[1].open(device, 1028, false), exposure.open(device, 16, false),
                               postData.open(device, postBytes, false)})
        if (!result)
            return result;
    return {};
}

std::expected<void, Error> RayReconstruction::open(const Settings& settings) {
    auto opened = std::make_unique<State>();
    if (auto result = opened->open(settings); !result)
        return result;
    state = std::move(opened);
    return {};
}

void RayReconstruction::State::initialize(VkCommandBuffer commands) const {
    for (const Vulkan::Image* entry : {&history[0], &history[1], &mip, &pooledMip, &feedback, &motion[0], &motion[1], &centreMotion, &farthestMotion, &guide,
                                       &candidateWeights, &shape, &exposureImages[0], &exposureImages[1]})
        entry->initialize(commands);
    for (const Vulkan::Buffer* entry : {&exposure, &postData})
        vkCmdFillBuffer(commands, entry->handle(), 0, VK_WHOLE_SIZE, 0);
    network.upload(commands);
}

std::expected<void, Error> RayReconstruction::record(VkCommandBuffer commands, const Textures& textures, const Evaluation& evaluation) {
    if (!state)
        return fail("the ray reconstruction is not open");
    State& s = *state;
    const std::array<std::tuple<const char*, const Texture*, VkFormat>, 9> bound{{
        {"color", &textures.color, VK_FORMAT_R16G16B16A16_SFLOAT},
        {"diffuseAlbedo", &textures.diffuseAlbedo, VK_FORMAT_R16G16B16A16_SFLOAT},
        {"specularAlbedo", &textures.specularAlbedo, VK_FORMAT_R16G16B16A16_SFLOAT},
        {"normals", &textures.normals, VK_FORMAT_R16G16B16A16_SFLOAT},
        {"roughness", &textures.roughness, VK_FORMAT_R16_SFLOAT},
        {"depth", &textures.depth, VK_FORMAT_R32_SFLOAT},
        {"motionVectors", &textures.motionVectors, VK_FORMAT_R16G16_SFLOAT},
        {"specularHitDistance", &textures.specularHitDistance, VK_FORMAT_R16_SFLOAT},
        {"output", &textures.output, VK_FORMAT_R16G16B16A16_SFLOAT},
    }};
    for (std::size_t index = 0; index < bound.size(); ++index) {
        const auto& [name, texture, format] = bound[index];
        if (!texture->image || !texture->view)
            return fail(std::format("the image {} is missing", name));
        const Size size = index + 1 == bound.size() ? s.outputSize : s.renderSize;
        s.inputs[index].borrow(texture->image, texture->view, format, size[0], size[1]);
    }
    const auto& [color, diffuseAlbedo, specularAlbedo, normals, roughness, depth, motionVectors, specularHitDistance, output] = s.inputs;

    //! Even frames (the first is frame 0) use the first image of each pair as the current one, odd frames the second.
    const std::uint32_t parity = s.frameIndex % 2;
    const std::uint32_t other = 1 - parity;
    const bool reset = evaluation.reset || s.frameIndex == 0;
    if (!s.initialized) {
        s.initialize(commands);
        s.initialized = true;
    }
    //! The frame index keeps counting through a reset.
    if (reset) {
        if (s.frameIndex == 0)
            for (const Vulkan::Buffer& sums : s.exposureSums)
                Kernels::recordClearRange(commands, sums, 0, sums.size());
        const VkClearColorValue one{.float32 = {1.0f, 0.0f, 0.0f, 0.0f}};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (const Vulkan::Image& image : s.exposureImages)
            vkCmdClearColorImage(commands, image.handle(), VK_IMAGE_LAYOUT_GENERAL, &one, 1, &range);
        Vulkan::fullBarrier(commands);
    } else {
        //! The histogram does not read the pooled mip: no barrier between them.
        s.pooledHistory.record(commands, s.mip, s.pooledMip);
    }
    s.histogram.record(commands, {color, s.exposureImages[parity], s.exposureImages[other], s.exposureSums[parity], s.exposureSums[other], s.exposure}, reset);
    Vulkan::fullBarrier(commands);

    //! The features of enc0: the camera, the size ratios and the constants of the weights.
    const CameraMatrices camera{evaluation.worldToView, evaluation.viewToClip};
    const std::array<std::uint32_t, cameraWordCount> words = getCameraWords(camera, reset ? camera : s.previousCamera, evaluation.jitter, s.renderSize);
    const std::array<float, 2> previousJitter = s.frameIndex == 0 ? std::array<float, 2>{} : s.previousJitter;
    const Size mipSize = halve(s.postSize);
    Network::FeatureParameters parameters{.viewToClip = matrix(words, 16), .pixelToWorld = matrix(words, 64), .worldToPreviousPixel = matrix(words, 80)};
    copy(parameters.size, s.network.size());
    copy(parameters.renderSize, s.renderSize);
    copy(parameters.jitter, evaluation.jitter);
    copy(parameters.previousJitter, previousJitter);
    copy(parameters.renderFromHistory, divide(s.renderSize, s.postSize));
    copy(parameters.historyFromRender, divide(s.postSize, s.renderSize));
    copy(parameters.inverseRenderSize, inverse(s.renderSize));
    copy(parameters.inverseHistorySize, inverse(s.postSize));
    copy(parameters.historySize, {static_cast<float>(s.postSize[0]), static_cast<float>(s.postSize[1])});
    copy(parameters.halfSize, {static_cast<float>(mipSize[0]), static_cast<float>(mipSize[1])});
    //! The mip covers 2·mipSize post pixels and the pooled mip 2·pooledSize mip pixels (both sizes round up).
    copy(parameters.inverseHalfSize, inverse({2 * mipSize[0], 2 * mipSize[1]}));
    copy(parameters.inverseQuarterSize, inverse({2 * halve(mipSize)[0], 2 * halve(mipSize)[1]}));
    copy(parameters.feedbackOffset, {previousJitter[0] * parameters.inverseRenderSize[0], previousJitter[1] * parameters.inverseRenderSize[1]});
    parameters.motionDifferenceGain = s.network.constant("enc0.motionDifferenceGain");
    parameters.motionDifferenceScale = s.network.constant("enc0.motionDifferenceScale");
    parameters.historyScale = s.network.constant("enc0.historyScale");
    parameters.feedbackScale = s.network.constant("enc0.feedbackScale");
    parameters.reset = reset;
    parameters.frame = s.frameIndex % 8;
    parameters.noise = s.network.noiseOffset();
    const Network::Frame frame{color,      diffuseAlbedo,      specularAlbedo, normals,     roughness,  depth,           motionVectors,    specularHitDistance,
                               s.exposure, s.motion[parity],   s.mip,          s.pooledMip, s.feedback, s.motion[other], s.farthestMotion, s.centreMotion,
                               s.guide,    s.candidateWeights, s.shape,        s.postData};
    s.network.record(commands, frame, parameters, parity);

    //! Post and downsample over the whole output, in one launch each.
    const Size level0 = s.network.size();
    const Size pitch = {(s.postSize[0] + 1) / 2 * 2, (s.postSize[1] + 1) / 2 * 2};
    const std::uint32_t colorOffset = 8 * level0[0] * level0[1];
    const std::uint32_t blueOffset = colorOffset + 4 * pitch[0] * pitch[1];
    Kernels::Post::Constants postConstants{};
    copy(postConstants.scale, divide(s.renderSize, s.postSize));
    copy(postConstants.offset, evaluation.jitter);
    copy(postConstants.outputSize, s.postSize);
    copy(postConstants.historyInverseSize, inverse(s.postSize));
    copy(postConstants.inputSize, s.renderSize);
    copy(postConstants.end, s.postSize);
    copy(postConstants.pitch, pitch);
    copy(postConstants.inputInverseSize, inverse(s.renderSize));
    postConstants.blendScale = std::exp(s.network.constant("post.logBlend"));
    postConstants.sourcePitch = static_cast<std::int32_t>(level0[0]);
    postConstants.colorOffset = colorOffset;
    postConstants.blueOffset = blueOffset;
    postConstants.reset = reset;
    s.post.record(commands,
                  {s.postData, s.exposure, s.guide, s.candidateWeights, s.shape, s.history[parity], s.motion[other], s.farthestMotion, s.centreMotion,
                   s.history[other], s.mip},
                  postConstants);
    Vulkan::fullBarrier(commands);

    const Size evenOutput = {(s.outputSize[0] + 1) / 2 * 2, (s.outputSize[1] + 1) / 2 * 2};
    Kernels::Downsample::Constants downsampleConstants{};
    copy(downsampleConstants.sourceSize, s.postSize);
    copy(downsampleConstants.outputSize, evenOutput);
    copy(downsampleConstants.end, evenOutput);
    copy(downsampleConstants.pitch, pitch);
    copy(downsampleConstants.regionEnd, s.postSize);
    downsampleConstants.colorOffset = colorOffset;
    downsampleConstants.blueOffset = blueOffset;
    s.downsample.record(commands, s.postData, output, s.exposure, downsampleConstants);
    Vulkan::fullBarrier(commands);

    s.previousJitter = evaluation.jitter;
    s.previousCamera = camera;
    ++s.frameIndex;
    return {};
}

}
