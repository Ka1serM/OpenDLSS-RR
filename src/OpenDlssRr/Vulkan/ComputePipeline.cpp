#include "OpenDlssRr/Vulkan/ComputePipeline.h"

#include <vector>

namespace OpenDlssRr::Vulkan {

ComputePipeline::~ComputePipeline() {
    if (pipeline)
        vkDestroyPipeline(device, pipeline, nullptr);
    if (layout)
        vkDestroyPipelineLayout(device, layout, nullptr);
    if (setLayout)
        vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
}

std::expected<void, Error> ComputePipeline::open(const Device& owner, const std::span<const std::byte> spirv, const char* entryPoint,
                                                 const std::span<const VkDescriptorType> descriptors, const std::uint32_t pushConstantBytes,
                                                 const std::span<const std::uint32_t> specialization, const std::uint32_t subgroupSize) {
    device = owner.handle();
    constantBytes = pushConstantBytes;
    types.assign(descriptors.begin(), descriptors.end());
    pushDescriptorSet = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"));
    if (!pushDescriptorSet)
        return fail("vkCmdPushDescriptorSetKHR is missing");

    if (spirv.empty() || spirv.size() % 4)
        return fail("the SPIR-V is not a whole number of words");

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (std::uint32_t index = 0; index < types.size(); ++index)
        bindings.push_back({.binding = index, .descriptorType = types[index], .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT});
    const VkDescriptorSetLayoutCreateInfo setInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                  .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                  .bindingCount = static_cast<std::uint32_t>(bindings.size()),
                                                  .pBindings = bindings.data()};
    if (auto result = check(vkCreateDescriptorSetLayout(device, &setInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout"); !result)
        return result;
    const VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = pushConstantBytes};
    const VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                .setLayoutCount = 1,
                                                .pSetLayouts = &setLayout,
                                                .pushConstantRangeCount = pushConstantBytes ? 1u : 0u,
                                                .pPushConstantRanges = &range};
    if (auto result = check(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout"); !result)
        return result;

    const VkShaderModuleCreateInfo moduleInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spirv.size(), .pCode = reinterpret_cast<const std::uint32_t*>(spirv.data())};
    VkShaderModule module = VK_NULL_HANDLE;
    if (auto result = check(vkCreateShaderModule(device, &moduleInfo, nullptr, &module), "vkCreateShaderModule"); !result)
        return result;
    std::vector<VkSpecializationMapEntry> entries;
    for (std::uint32_t index = 0; index < specialization.size(); ++index)
        entries.push_back({.constantID = index, .offset = index * 4, .size = 4});
    const VkSpecializationInfo specializationInfo{.mapEntryCount = static_cast<std::uint32_t>(entries.size()),
                                                  .pMapEntries = entries.data(),
                                                  .dataSize = specialization.size_bytes(),
                                                  .pData = specialization.data()};
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo requiredSize{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
                                                                           .requiredSubgroupSize = subgroupSize};
    const VkComputePipelineCreateInfo pipelineInfo{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                   .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                             .pNext = subgroupSize ? &requiredSize : nullptr,
                                                             .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                             .module = module,
                                                             .pName = entryPoint,
                                                             .pSpecializationInfo = specialization.empty() ? nullptr : &specializationInfo},
                                                   .layout = layout};
    std::expected<void, Error> result =
        check(vkCreateComputePipelines(device, owner.pipelineCache(), 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");
    vkDestroyShaderModule(device, module, nullptr);
    return result;
}

void ComputePipeline::dispatch(VkCommandBuffer commands, const std::span<const Binding> bindings, const void* pushConstants, const std::uint32_t x,
                               const std::uint32_t y, const std::uint32_t z) const {
    std::vector<VkDescriptorImageInfo> images(bindings.size());
    std::vector<VkDescriptorBufferInfo> buffers(bindings.size());
    std::vector<VkWriteDescriptorSet> writes;
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = index, .descriptorCount = 1, .descriptorType = bindings[index].type};
        if (bindings[index].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || bindings[index].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            buffers[index] = {bindings[index].buffer, 0, VK_WHOLE_SIZE};
            write.pBufferInfo = &buffers[index];
        } else {
            images[index] = {bindings[index].sampler, bindings[index].view, VK_IMAGE_LAYOUT_GENERAL};
            write.pImageInfo = &images[index];
        }
        writes.push_back(write);
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (!writes.empty())
        pushDescriptorSet(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, static_cast<std::uint32_t>(writes.size()), writes.data());
    if (constantBytes)
        vkCmdPushConstants(commands, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constantBytes, pushConstants);
    vkCmdDispatch(commands, x, y, z);
}

}
