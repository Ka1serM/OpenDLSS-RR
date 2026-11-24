#include "OpenDlssRr/Network.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <set>
#include <span>
#include <tuple>
#include <vector>

namespace OpenDlssRr {

namespace {

using Vulkan::Binding;
using Vulkan::fail;

constexpr VkDescriptorType storageBuffer = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
constexpr VkDescriptorType sampledImage = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
constexpr VkDescriptorType filteredImage = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
constexpr VkDescriptorType storageImage = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

//! The weight buffer of each kernel (docs/weights.md); enc0 has the stem in b06 and its block in b08.
const std::map<std::string_view, std::string, std::less<>> tensorBuffers{
    {"enc0", "b08"},   {"enc1", "b10"},   {"enc2", "b12"},   {"enc3", "b14"},   {"enc4", "b16"},   {"enc5", "b18"},   {"dec4_0", "b20"}, {"dec4_1", "b22"},
    {"dec3_0", "b24"}, {"dec3_1", "b26"}, {"dec2_0", "b28"}, {"dec2_1", "b30"}, {"dec1_0", "b32"}, {"dec1_1", "b34"}, {"dec0_0", "b36"}, {"dec0_1", "b38"}};

//! The down convolutions from each level to the next coarser one.
const std::set<std::string, std::less<>> downConvolutions{"b10.t00", "b10.t10", "b12.t08", "b14.t08", "b16.t08"};

//! The largest output of a linear layer without a bias (the QKV projection of L4 and L5).
constexpr std::uint32_t zeroCount = 512;

//! FP32 to FP16 bits, round to nearest even; the values are finite and within the FP16 range.
std::uint16_t floatToHalf(const float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x38800000u)
        return static_cast<std::uint16_t>(sign | ((magnitude + 0x0fffu + ((magnitude >> 13) & 1u) - 0x38000000u) >> 13));
    const float subnormal = std::bit_cast<float>(magnitude) * 16777216.0f;
    return static_cast<std::uint16_t>(sign | static_cast<std::uint32_t>(std::nearbyint(subnormal)));
}

std::string tensor(const std::string& buffer, const std::uint32_t index) {
    return std::format("{}.t{:02d}", buffer, index);
}

Binding buffer(const Vulkan::Buffer& source) {
    return {.type = storageBuffer, .buffer = source.handle()};
}

Binding storage(const Vulkan::Image& image) {
    return Binding{.type = storageImage, .view = image.view()};
}

//! The row-major window index (8 y + x) of a token of the block kernel, which orders the tokens by 2x2 parity, then by 4x4 position.
std::size_t windowToken(const std::size_t token) {
    const std::size_t parity = token / 16;
    const std::size_t index = token % 16;
    return (2 * (index / 4) + parity / 2) * 8 + 2 * (index % 4) + parity % 2;
}

//! Encoders below L5 alternate (2, 6) and (6, 2); enc5 and decL_0 (4, 0) and (0, 4); decL_1 the other way round.
std::array<std::int32_t, 2> windowShift(const std::string_view kernel, const std::uint32_t parity) {
    if (kernel.starts_with("enc") && kernel != "enc5")
        return parity ? std::array{6, 2} : std::array{2, 6};
    const bool flipped = kernel.ends_with("_1") != (parity == 1);
    return flipped ? std::array{0, 4} : std::array{4, 0};
}

}

float Network::constant(const std::string_view name) const {
    const auto found = scalars.find(name);
    return found == scalars.end() ? 0.0f : found->second;
}

std::expected<void, Vulkan::Error> Network::open(const Vulkan::Device& device, const Weights& weights, const ShaderCode& shaders,
                                                 const std::array<std::uint32_t, 2> renderSize) {
    width = (renderSize[0] + 31) / 32 * 32;
    height = (renderSize[1] + 31) / 32 * 32;
    scalars = weights.constants;

    //! The pair tables become (M + Mᵀ) · pairScale in block token order, b06.t00 its 7-bit thresholds; the rest is also FP16, 16-byte aligned.
    std::vector<float> values;
    std::vector<std::uint16_t> halves;
    const float pairScale = weights.pairScale[0] * weights.pairScale[1];
    for (const auto& [name, source] : weights.tensors) {
        values.resize((values.size() + 7) / 8 * 8, 0.0f);
        halves.resize(values.size(), 0);
        offsets[name] = static_cast<std::uint32_t>(values.size());
        const auto at = [&](const std::size_t index) { return halfToFloat(source.values[index]); };
        if (name == "b06.t00")
            for (const std::uint16_t bits : source.values)
                values.push_back(static_cast<float>(bits >> 9));
        else if (source.shape.size() == 3 && source.shape[1] == 64 && source.shape[2] == 64)
            for (std::size_t head = 0; head < source.shape[0]; ++head)
                for (std::size_t i = 0; i < 64; ++i)
                    for (std::size_t j = 0; j < 64; ++j) {
                        const std::size_t a = windowToken(i);
                        const std::size_t b = windowToken(j);
                        values.push_back((at((head * 64 + a) * 64 + b) + at((head * 64 + b) * 64 + a)) * pairScale);
                        halves.push_back(floatToHalf(values.back()));
                    }
        else if (source.shape.size() == 2 && source.shape[1] % 32 != 0) {
            //! A matrix with rows that are not a multiple of 32 long (the stem, 43 inputs) has FP16 rows padded with zeros to a multiple of 32.
            const std::size_t stride = (source.shape[1] + 31) / 32 * 32;
            for (std::size_t index = 0; index < source.values.size(); ++index)
                values.push_back(at(index));
            halves.resize(offsets[name] + source.shape[0] * stride, 0);
            for (std::size_t row = 0; row < source.shape[0]; ++row)
                std::copy_n(source.values.begin() + static_cast<std::ptrdiff_t>(row * source.shape[1]), source.shape[1],
                            halves.begin() + static_cast<std::ptrdiff_t>(offsets[name] + row * stride));
            values.resize(halves.size(), 0.0f);
        } else if (source.shape.size() == 2 && source.shape[0] < 16 && source.shape[1] % 16 == 0) {
            //! A matrix with fewer than 16 rows (a colour head, 3 or 4 outputs) has FP16 rows padded with zero rows to 16.
            for (std::size_t index = 0; index < source.values.size(); ++index)
                values.push_back(at(index));
            halves.insert(halves.end(), source.values.begin(), source.values.end());
            values.resize(offsets[name] + 16 * source.shape[1], 0.0f);
        } else if (downConvolutions.contains(name)) {
            //! The FP16 copy of a down convolution is outputChannels x 4 x inputChannels, so that a row of A is 4 whole pixels.
            const std::size_t outputs = source.shape[0];
            const std::size_t inputs = source.values.size() / outputs / 4;
            for (std::size_t index = 0; index < source.values.size(); ++index)
                values.push_back(at(index));
            halves.resize(values.size(), 0);
            for (std::size_t o = 0; o < outputs; ++o)
                for (std::size_t c = 0; c < inputs; ++c)
                    for (std::size_t position = 0; position < 4; ++position)
                        halves[offsets[name] + (o * 4 + position) * inputs + c] = source.values[(o * inputs + c) * 4 + position];
        } else {
            for (std::size_t index = 0; index < source.values.size(); ++index)
                values.push_back(at(index));
            halves.insert(halves.end(), source.values.begin(), source.values.end());
        }
        halves.resize(values.size(), 0);
        //! Matrices are stored in 16x16 tiles of 512 contiguous bytes, so a cooperative matrix load reads one block.
        const std::size_t rows = source.shape.size() == 3 && source.shape[1] == 64 && source.shape[2] == 64 ? source.shape[0] * 64
                                 : source.shape.size() == 2 && source.shape[0] < 16                         ? 16
                                                                                                            : source.shape[0];
        const std::size_t count = halves.size() - offsets[name];
        if (name == "b06.t00" || source.shape.size() < 2 || rows % 16 != 0 || count % rows != 0 || count / rows % 16 != 0)
            continue;
        const std::size_t columns = count / rows;
        const std::vector<std::uint16_t> rowMajor(halves.begin() + offsets[name], halves.end());
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < columns; ++column)
                halves[offsets[name] + (row / 16 * (columns / 16) + column / 16) * 256 + row % 16 * 16 + column % 16] = rowMajor[row * columns + column];
    }
    offsets["zeros"] = static_cast<std::uint32_t>(values.size());
    values.resize(values.size() + zeroCount, 0.0f);
    const VkDeviceSize bytes = values.size() * sizeof(float);
    if (auto result = staging.open(device, bytes, true); !result)
        return result;
    std::memcpy(staging.mapped(), values.data(), bytes);
    if (auto result = weightArray.open(device, bytes, false); !result)
        return result;
    halves.resize(values.size(), 0);
    const VkDeviceSize halfBytes = halves.size() * sizeof(std::uint16_t);
    if (auto result = matrixStaging.open(device, halfBytes, true); !result)
        return result;
    std::memcpy(matrixStaging.mapped(), halves.data(), halfBytes);
    if (auto result = matrixArray.open(device, halfBytes, false); !result)
        return result;
    if (auto result = parameterBuffer.open(device, (sizeof(FeatureParameters) + 15) / 16 * 16, false); !result)
        return result;

    //! Each buffer has room for a multiple of 64 pixels: the MLP of a split block reads and writes whole tiles of 64.
    const auto allocate = [&](Vulkan::Buffer& target, const std::uint64_t halves) { return target.open(device, halves * 2, false); };
    std::uint64_t wide = 0;
    std::uint64_t blockInput = 0;
    std::uint64_t attended = 0;
    for (std::uint32_t level = 0; level < levels; ++level) {
        const std::uint64_t count = (pixels(level) + 63) / 64 * 64;
        wide = std::max<std::uint64_t>(wide, channels[level] * count);
        if (split(level)) {
            blockInput = std::max<std::uint64_t>(blockInput, channels[level] * count);
            attended = std::max<std::uint64_t>(attended, heads[level] * std::uint64_t{32} * count);
        }
        if (auto result = allocate(activations.encoded[level], channels[level] * count); !result)
            return result;
        if (auto result = allocate(activations.gated[level], 4 * count); !result)
            return result;
        if (level + 1 == levels)
            continue;
        if (auto result = allocate(activations.blended[level], 4 * count); !result)
            return result;
        //! dec0_1 ends with the output heads in place of its output, so L0 has no decoded buffer.
        if (level == 0)
            continue;
        if (auto result = allocate(activations.decoded[level], channels[level] * count); !result)
            return result;
    }
    for (auto [target, halves] : {std::pair{&activations.wide, wide}, std::pair{&activations.blockInput, std::max<std::uint64_t>(blockInput, 64)},
                                  std::pair{&activations.attended, std::max<std::uint64_t>(attended, 64)}})
        if (auto result = allocate(*target, halves); !result)
            return result;

    if (auto result = linearSampler.open(device, VK_FILTER_LINEAR); !result)
        return result;
    const std::array mlpDescriptors{storageBuffer, storageBuffer, storageBuffer, storageBuffer, storageBuffer, storageBuffer, storageBuffer, storageBuffer};
    //! The block bindings, then for the stem those of Shaders/Features.slang.
    const std::array blockDescriptors{storageBuffer, storageBuffer, storageBuffer, storageBuffer, storageBuffer, storageBuffer, storageBuffer,
                                      storageBuffer, storageImage,  storageImage,  storageImage,  storageImage,  storageBuffer};
    const std::array stemDescriptors{storageBuffer, storageBuffer, storageBuffer, storageBuffer, storageBuffer, storageBuffer,
                                     storageBuffer, storageBuffer, storageImage,  storageImage,  storageImage,  storageImage,
                                     storageBuffer, sampledImage,  sampledImage,  sampledImage,  sampledImage,  sampledImage,
                                     sampledImage,  sampledImage,  sampledImage,  filteredImage, filteredImage, filteredImage,
                                     filteredImage, storageBuffer, storageImage,  storageImage,  storageImage,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};

    //! The block kernel needs subgroups of 32; the size is required only where the default is different (it needs subgroupSizeControl).
    VkPhysicalDeviceSubgroupProperties subgroup{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &subgroup};
    vkGetPhysicalDeviceProperties2(device.physical(), &properties);
    const std::uint32_t blockSubgroupSize = subgroup.subgroupSize == 32 ? 0 : 32;
    //! The blocks of enc0 ... enc3 end with the down convolution of the next encoder, so enc1 ... enc4 begin with it done.
    for (std::uint32_t level = 0; level < levels; ++level) {
        //! If 4 does not divide the column tiles, 2 groups of 2 subgroups split the rows too, so each subgroup has as many tiles.
        const std::uint32_t tiles = channels[level] / 16;
        const std::uint32_t rowGroups = tiles % 4 == 0 ? 1 : 2;
        const std::uint32_t columnGroups = 4 / rowGroups;
        const std::uint32_t columns = (tiles + columnGroups - 1) / columnGroups;
        const std::array<std::pair<Prologue, std::uint32_t>, 4> prologues{{{Prologue::None, 0},
                                                                           {level == 0   ? Prologue::Stem
                                                                            : level == 5 ? Prologue::Down
                                                                                         : Prologue::Downsampled,
                                                                            level ? channels[level - 1] : 0},
                                                                           {Prologue::Up, level + 1 < levels ? channels[level + 1] : 0}}};
        for (const auto& [prologue, sourceChannels] : prologues) {
            if ((prologue == Prologue::Up && level + 1 == levels) || (prologue == Prologue::Downsampled && split(level)))
                continue;
            const Stage stage = !split(level) ? Stage::Block : prologue == Prologue::None ? Stage::Attention : Stage::Input;
            //! Stage Input splits the column tiles into parts of 2 (inputTiles), one part per workgroup and one row tile per subgroup.
            const bool input = stage == Stage::Input;
            const bool encoder = prologue == Prologue::Stem || prologue == Prologue::Downsampled;
            const std::uint32_t nextChannels = encoder && level + 1 < levels ? channels[level + 1] : 0;
            const std::array specialization{channels[level],
                                            heads[level],
                                            input ? 4 : rowGroups,
                                            input ? inputTiles : columns,
                                            static_cast<std::uint32_t>(prologue),
                                            sourceChannels,
                                            static_cast<std::uint32_t>(stage),
                                            static_cast<std::uint32_t>(split(level) ? Head::None : head(level, prologue)),
                                            nextChannels};
            const bool stem = prologue == Prologue::Stem;
            if (auto result = blockPasses[{level, prologue, stage}].open(device, stem ? shaders.stem : shaders.block, stem ? "stem" : "block",
                                                                         stem ? std::span<const VkDescriptorType>{stemDescriptors} : blockDescriptors, 108,
                                                                         specialization, blockSubgroupSize);
                !result)
                return result;
        }
        if (!split(level))
            continue;
        //! The MLP of a split block ends with its colour head.
        for (const Head colourHead : {Head::None, Head::GatePooled, Head::Blend}) {
            const std::array specialization{channels[level], heads[level], static_cast<std::uint32_t>(colourHead)};
            if (auto result = mlpPasses[{level, colourHead}].open(device, shaders.mlp, "mlp", mlpDescriptors, 108, specialization, blockSubgroupSize); !result)
                return result;
        }
    }
    return {};
}

void Network::upload(VkCommandBuffer commands) const {
    const VkBufferCopy region{.size = staging.size()};
    vkCmdCopyBuffer(commands, staging.handle(), weightArray.handle(), 1, &region);
    const VkBufferCopy matrixRegion{.size = matrixStaging.size()};
    vkCmdCopyBuffer(commands, matrixStaging.handle(), matrixArray.handle(), 1, &matrixRegion);
    Vulkan::fullBarrier(commands);
}

Network::Head Network::head(const std::uint32_t level, const Prologue prologue) {
    if (prologue == Prologue::Up)
        return Head::None;
    if (prologue == Prologue::None)
        return level == 0 ? Head::Output : Head::Blend;
    return level == 0 ? Head::Gate : Head::GatePooled;
}

void Network::block(VkCommandBuffer commands, const Frame& frame, const std::uint32_t level, const Vulkan::Buffer& input, const Vulkan::Buffer& target,
                    const std::string& tensors, const std::uint32_t first, const std::array<std::int32_t, 2> shift, const std::string_view kernel,
                    const BlockInput& prologue, const ColourHead& colourHead) const {
    const std::string name(kernel);
    struct {
        std::int32_t size[2], shift[2];
        std::uint32_t qkv, pairs, projection, projectionBias, hidden, hiddenBias, mlp, mlpBias;
        float attentionScale, mlpScale;
        std::uint32_t prologueWeight, prologueBias, stemOutput, stemOutputBias;
        float skipScale;
        std::uint32_t headWeight, headBias;
        float gain;
        std::int32_t renderSize[2];
        float correlationScale;
        std::uint32_t nextWeight, nextBias;
    } constants{{static_cast<std::int32_t>(size(level)[0]), static_cast<std::int32_t>(size(level)[1])},
                {shift[0], shift[1]},
                offset(tensor(tensors, first)),
                offset(tensor(tensors, first + 1)),
                offset(tensor(tensors, first + 2)),
                offset(tensor(tensors, first + 3)),
                offset(tensor(tensors, first + 4)),
                offset(tensor(tensors, first + 5)),
                offset(tensor(tensors, first + 6)),
                offset(tensor(tensors, first + 7)),
                constant(name + ".attention"),
                constant(name + ".mlp"),
                prologue.weight.empty() ? 0 : offset(prologue.weight),
                prologue.bias.empty() ? 0 : offset(prologue.bias),
                prologue.prologue == Prologue::Stem ? offset("b06.t03") : 0,
                prologue.prologue == Prologue::Stem ? offset("b06.t04") : 0,
                prologue.skipScale,
                colourHead.weight.empty() ? 0 : offset(colourHead.weight),
                colourHead.bias.empty() ? 0 : offset(colourHead.bias),
                colourHead.gain,
                {colourHead.renderSize[0], colourHead.renderSize[1]},
                colourHead.correlationScale,
                colourHead.next ? offset(colourHead.nextWeight) : 0,
                colourHead.next ? offset(colourHead.nextBias) : 0};
    const auto windows = [&](const std::uint32_t extent, const std::int32_t offset) { return (extent + static_cast<std::uint32_t>(offset) + 7) / 8; };
    const std::uint32_t windowsX = windows(size(level)[0], shift[0]);
    const std::uint32_t windowsY = windows(size(level)[1], shift[1]);
    const Vulkan::Buffer& source = prologue.source ? *prologue.source : input;
    if (!split(level)) {
        std::vector<Binding> bindings{buffer(input),
                                      buffer(matrixArray),
                                      buffer(weightArray),
                                      buffer(target),
                                      buffer(source),
                                      buffer(colourHead.colour ? *colourHead.colour : input),
                                      buffer(colourHead.coarse ? *colourHead.coarse : input),
                                      buffer(colourHead.output ? *colourHead.output : target),
                                      storage(frame.feedback),
                                      storage(frame.guide),
                                      storage(frame.candidateWeights),
                                      storage(frame.shape),
                                      buffer(colourHead.next ? *colourHead.next : target)};
        if (prologue.prologue == Prologue::Stem) {
            const auto sampled = [](const Vulkan::Image& image) { return Binding{.type = sampledImage, .view = image.view()}; };
            const auto filtered = [this](const Vulkan::Image& image) {
                return Binding{.type = filteredImage, .view = image.view(), .sampler = linearSampler.handle()};
            };
            const auto raw = [](const Vulkan::Image& image) { return Binding{.type = storageImage, .view = image.rawView()}; };
            bindings.insert(bindings.end(), {sampled(frame.color), sampled(frame.diffuseAlbedo), sampled(frame.specularAlbedo), sampled(frame.normals),
                                             sampled(frame.roughness), sampled(frame.depth), sampled(frame.motionVectors), sampled(frame.specularHitDistance),
                                             filtered(frame.previousMotion), filtered(frame.history), filtered(frame.pooledHistory), filtered(frame.feedback),
                                             buffer(frame.exposure), storage(frame.motion), raw(frame.farthestMotion), raw(frame.centreMotion),
                                             Binding{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .buffer = parameterBuffer.handle()}});
        }
        blockPasses.at({level, prologue.prologue, Stage::Block}).dispatch(commands, bindings, &constants, windowsX, windowsY, 1);
        Vulkan::fullBarrier(commands);
        return;
    }
    const bool ready = prologue.prologue == Prologue::None || prologue.prologue == Prologue::Downsampled;
    const Vulkan::Buffer& x = ready ? input : activations.blockInput;
    if (!ready) {
        const std::array bindings{buffer(input),
                                  buffer(matrixArray),
                                  buffer(weightArray),
                                  buffer(activations.blockInput),
                                  buffer(source),
                                  buffer(input),
                                  buffer(input),
                                  buffer(activations.blockInput),
                                  storage(frame.feedback),
                                  storage(frame.guide),
                                  storage(frame.candidateWeights),
                                  storage(frame.shape),
                                  buffer(input)};
        blockPasses.at({level, prologue.prologue, Stage::Input})
            .dispatch(commands, bindings, &constants, windowsX, windowsY, channels[level] / 16 / inputTiles);
        Vulkan::fullBarrier(commands);
    }
    const std::array attention{buffer(x),
                               buffer(matrixArray),
                               buffer(weightArray),
                               buffer(activations.attended),
                               buffer(x),
                               buffer(x),
                               buffer(x),
                               buffer(activations.attended),
                               storage(frame.feedback),
                               storage(frame.guide),
                               storage(frame.candidateWeights),
                               storage(frame.shape),
                               buffer(x)};
    blockPasses.at({level, Prologue::None, Stage::Attention}).dispatch(commands, attention, &constants, windowsX, windowsY, heads[level]);
    Vulkan::fullBarrier(commands);
    const std::array mlp{buffer(x),
                         buffer(matrixArray),
                         buffer(weightArray),
                         buffer(target),
                         buffer(activations.attended),
                         buffer(colourHead.colour ? *colourHead.colour : x),
                         buffer(colourHead.coarse ? *colourHead.coarse : x),
                         buffer(colourHead.output ? *colourHead.output : target)};
    mlpPasses.at({level, colourHead.output ? head(level, prologue.prologue) : Head::None}).dispatch(commands, mlp, &constants, (pixels(level) + 15) / 16, 1, 1);
    Vulkan::fullBarrier(commands);
}

void Network::record(VkCommandBuffer commands, const Frame& frame, const FeatureParameters& parameters, const std::uint32_t parity,
                     const std::uint32_t kernelCount) const {
    const Buffers& a = activations;

    //! The blocks of enc0 ... enc3 write the down convolution to the decoded buffer of the next level, which is free until the decoder.
    const auto down = [&](const std::uint32_t level) {
        const std::string& tensors = tensorBuffers.at(level == 1 ? "enc1" : "enc" + std::to_string(level - 1));
        const std::uint32_t first = level == 1 ? 0 : (level == 2 ? 2 : 0) + 8;
        return std::pair{tensor(tensors, first), tensor(tensors, first + 1)};
    };
    const auto withNext = [&](ColourHead colourHead, const std::uint32_t level) {
        if (level + 1 >= levels || split(level))
            return colourHead;
        colourHead.next = &a.decoded[level + 1];
        std::tie(colourHead.nextWeight, colourHead.nextBias) = down(level + 1);
        return colourHead;
    };

    for (std::uint32_t index = 0; index < kernelCount && index < kernels.size(); ++index) {
        const std::string_view kernel = kernels[index];
        const std::string& tensors = tensorBuffers.at(kernel);
        const std::array<std::int32_t, 2> shift = windowShift(kernel, parity);
        if (kernel == "enc0") {
            //! The features of the stem read the parameters (Shaders/Features.slang).
            vkCmdUpdateBuffer(commands, parameterBuffer.handle(), 0, sizeof(parameters), &parameters);
            Vulkan::fullBarrier(commands);
            //! The block begins with the stem 43 -> 64 -> 32 with a GELU; then the colour head.
            block(commands, frame, 0, a.wide, a.encoded[0], tensors, 0, shift, kernel, {.prologue = Prologue::Stem, .weight = "b06.t01", .bias = "b06.t02"},
                  withNext({.output = &a.gated[0], .weight = tensor(tensors, 8), .bias = tensor(tensors, 9), .gain = constant(std::string(kernel) + ".gain")},
                           0));
        } else if (kernel.starts_with("enc")) {
            //! The block begins with the down convolution of the finer level, which the finer block did (enc1 ... enc4) or the prologue does.
            const std::uint32_t level = static_cast<std::uint32_t>(kernel[3] - '0');
            const std::uint32_t first = level == 1 ? 2 : 0;
            const std::uint32_t head = level == 5 ? first + 8 : first + 10;
            const auto [weight, bias] = down(level);
            const BlockInput input = level == 5 ? BlockInput{.prologue = Prologue::Down, .source = &a.encoded[level - 1], .weight = weight, .bias = bias}
                                                : BlockInput{.prologue = Prologue::Downsampled};
            block(commands, frame, level, level == 5 ? a.encoded[level - 1] : a.decoded[level], a.encoded[level], tensors, first, shift, kernel, input,
                  withNext({.colour = &a.gated[level - 1],
                            .output = &a.gated[level],
                            .weight = tensor(tensors, head),
                            .bias = tensor(tensors, head + 1),
                            .gain = constant(std::string(kernel) + ".gain")},
                           level));
        } else if (kernel.ends_with("_0")) {
            //! The second kernel of the level reads this block output from the scratch buffer wide.
            const std::uint32_t level = static_cast<std::uint32_t>(kernel[3] - '0');
            const Vulkan::Buffer& coarse = level == 4 ? a.encoded[5] : a.decoded[level + 1];
            block(commands, frame, level, a.encoded[level], a.wide, tensors, 2, shift, kernel,
                  {.prologue = Prologue::Up,
                   .source = &coarse,
                   .weight = tensor(tensors, 0),
                   .bias = tensor(tensors, 1),
                   .skipScale = constant(std::string(kernel) + ".skip")},
                  ColourHead{});
        } else {
            //! A block, then the colour head: Blend on levels 1 ... 4, the output heads on L0.
            const std::uint32_t level = static_cast<std::uint32_t>(kernel[3] - '0');
            const Vulkan::Buffer& coarse = level == 4 ? a.gated[5] : a.blended[level + 1];
            ColourHead colourHead;
            if (level > 0)
                colourHead = {.colour = &a.gated[level],
                              .coarse = &coarse,
                              .output = &a.blended[level],
                              .weight = tensor(tensors, 8),
                              .bias = tensor(tensors, 9),
                              .gain = constant(std::string(kernel) + ".gain")};
            else
                colourHead = {.colour = &a.gated[0],
                              .coarse = &coarse,
                              .output = &frame.colour,
                              .gain = constant("dec0_1.gain"),
                              .renderSize = {parameters.renderSize[0], parameters.renderSize[1]},
                              .correlationScale = std::exp(constant("dec0_1.logCorrelation"))};
            block(commands, frame, level, a.wide, level ? a.decoded[level] : a.wide, tensors, 0, shift, kernel, BlockInput{}, colourHead);
        }
    }
}

}
