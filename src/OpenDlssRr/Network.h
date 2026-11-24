#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <map>
#include <string>
#include <string_view>
#include <tuple>

#include "OpenDlssRr/ShaderCode.h"
#include "OpenDlssRr/Vulkan/ComputePipeline.h"
#include "OpenDlssRr/Vulkan/Resources.h"
#include "OpenDlssRr/Weights.h"

namespace OpenDlssRr {

//! The network (docs/network.md) as compute passes. Activations are FP16 buffers in the layout [y][x][channel]; the maths is FP32.
class Network {
public:
    static constexpr std::uint32_t levels = 6;
    static constexpr std::array<std::uint32_t, levels> channels{32, 64, 96, 128, 160, 160};
    static constexpr std::array<std::uint32_t, levels> heads{2, 2, 4, 4, 8, 8};
    //! The network kernels, in order; record() can stop after any of them.
    static constexpr std::array<std::string_view, 16> kernels{"enc0",   "enc1",   "enc2",   "enc3",   "enc4",   "enc5",   "dec4_0", "dec4_1",
                                                              "dec3_0", "dec3_1", "dec2_0", "dec2_1", "dec1_0", "dec1_1", "dec0_0", "dec0_1"};

    //! The uniform block of the features pass (Shaders/Features.slang, std140).
    struct FeatureParameters {
        std::array<float, 16> viewToClip;
        std::array<float, 16> pixelToWorld;
        std::array<float, 16> worldToPreviousPixel;
        std::int32_t size[2];
        std::int32_t renderSize[2];
        float jitter[2];
        float previousJitter[2];
        float renderFromHistory[2];
        float historyFromRender[2];
        float inverseRenderSize[2];
        float inverseHistorySize[2];
        float historySize[2];
        float halfSize[2];
        float inverseHalfSize[2];
        float inverseQuarterSize[2];
        float feedbackOffset[2];
        float motionDifferenceGain;
        float motionDifferenceScale;
        float historyScale;
        float feedbackScale;
        std::uint32_t reset;
        std::uint32_t frame;
        std::uint32_t noise;
    };

    //! The images and buffers of one frame. The network reads the inputs and the state of the last frame, and writes the rest.
    struct Frame {
        const Vulkan::Image& color;
        const Vulkan::Image& diffuseAlbedo;
        const Vulkan::Image& specularAlbedo;
        const Vulkan::Image& normals;
        const Vulkan::Image& roughness;
        const Vulkan::Image& depth;
        const Vulkan::Image& motionVectors;
        const Vulkan::Image& specularHitDistance;
        //! The exposure of HistogramBasic.
        const Vulkan::Buffer& exposure;
        //! The state of the last frame, which this frame then overwrites.
        const Vulkan::Image& previousMotion;
        const Vulkan::Image& history;
        const Vulkan::Image& pooledHistory;
        const Vulkan::Image& feedback;
        //! Render size: nearest and reflection motion (RGBA16F), farthest and centre motion (RG16F with an R32_UINT raw view).
        const Vulkan::Image& motion;
        const Vulkan::Image& farthestMotion;
        const Vulkan::Image& centreMotion;
        //! Render size: the post filter inputs.
        const Vulkan::Image& guide;
        const Vulkan::Image& candidateWeights;
        const Vulkan::Image& shape;
        //! The colour that post reads: FP16 RGBA per pixel of the L0 size, at offset 0.
        const Vulkan::Buffer& colour;
    };

    //! Unpacks the weights into one FP32 array and creates the buffers for a render size and the pipelines.
    [[nodiscard]] std::expected<void, Vulkan::Error> open(const Vulkan::Device& device, const Weights& weights, const ShaderCode& shaders,
                                                          std::array<std::uint32_t, 2> renderSize);
    //! Copies the weights to the GPU; once, before the first record().
    void upload(VkCommandBuffer commands) const;
    //! Records the passes of the first `kernelCount` kernels, with a full barrier after each pass. parity is the frame index modulo 2.
    void record(VkCommandBuffer commands, const Frame& frame, const FeatureParameters& parameters, std::uint32_t parity,
                std::uint32_t kernelCount = kernels.size()) const;

    //! The L0 size: the render size rounded up to 32. Level l is 2^l times smaller.
    [[nodiscard]] std::array<std::uint32_t, 2> size(std::uint32_t level = 0) const { return {width >> level, height >> level}; }
    [[nodiscard]] float constant(std::string_view name) const;
    [[nodiscard]] std::uint32_t noiseOffset() const { return offsets.at("b06.t00"); }

    //! The activations and colours of each level, for tests.
    struct Buffers {
        std::array<Vulkan::Buffer, levels> encoded;
        std::array<Vulkan::Buffer, levels - 1> decoded;
        std::array<Vulkan::Buffer, levels> gated;
        std::array<Vulkan::Buffer, levels - 1> blended;
        //! Scratch: the output of the first block of a decoder.
        Vulkan::Buffer wide;
        //! Scratch of the split blocks: the block input after the prologue, and the attention output (32 channels per head).
        Vulkan::Buffer blockInput;
        Vulkan::Buffer attended;
    };
    [[nodiscard]] const Buffers& buffers() const { return activations; }

private:
    //! Downsampled: the finer encoder block already computed the down convolution after its output.
    enum class Prologue : std::uint32_t { None, Stem, Down, Up, Downsampled };
    struct BlockInput {
        Prologue prologue = Prologue::None;
        const Vulkan::Buffer* source = nullptr;
        std::string weight;
        std::string bias;
        float skipScale = 0.0f;
    };
    enum class Head : std::uint32_t { None, Gate, GatePooled, Blend, Output };
    struct ColourHead {
        const Vulkan::Buffer* colour = nullptr;
        const Vulkan::Buffer* coarse = nullptr;
        const Vulkan::Buffer* output = nullptr;
        std::string weight;
        std::string bias;
        float gain = 0.0f;
        std::array<std::int32_t, 2> renderSize{};
        float correlationScale = 0.0f;
        //! The down convolution of the next encoder after the block, to this buffer.
        const Vulkan::Buffer* next = nullptr;
        std::string nextWeight;
        std::string nextBias;
    };
    [[nodiscard]] static Head head(std::uint32_t level, Prologue prologue);
    void block(VkCommandBuffer commands, const Frame& frame, std::uint32_t level, const Vulkan::Buffer& input, const Vulkan::Buffer& target,
               const std::string& tensors, std::uint32_t first, std::array<std::int32_t, 2> shift, std::string_view kernel, const BlockInput& prologue,
               const ColourHead& colourHead) const;
    [[nodiscard]] std::uint32_t pixels(std::uint32_t level) const { return (width >> level) * (height >> level); }
    //! The levels with few windows split a block into prologue, attention per head and Shaders/Mlp.slang dispatches.
    [[nodiscard]] static bool split(std::uint32_t level) { return level >= 4; }
    static constexpr std::uint32_t inputTiles = 2;
    [[nodiscard]] std::uint32_t offset(const std::string& tensor) const { return offsets.at(tensor); }

    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::map<std::string, std::uint32_t, std::less<>> offsets;
    std::map<std::string, float, std::less<>> scalars;
    Vulkan::Buffer staging;
    Vulkan::Buffer weightArray;
    Vulkan::Buffer matrixStaging;
    Vulkan::Buffer matrixArray;
    Vulkan::Buffer parameterBuffer;
    Buffers activations;
    Vulkan::Sampler linearSampler;
    //! The stage of a block pipeline (Shaders/Block.slang): the whole block, only the block input, or only the attention of one head.
    enum class Stage : std::uint32_t { Block, Input, Attention };
    //! The block pipelines by level, prologue and stage, and the MLP pipelines of the split levels.
    std::map<std::tuple<std::uint32_t, Prologue, Stage>, Vulkan::ComputePipeline> blockPasses;
    std::map<std::pair<std::uint32_t, Head>, Vulkan::ComputePipeline> mlpPasses;
};

}
