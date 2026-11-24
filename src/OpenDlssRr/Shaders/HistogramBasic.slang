//! Exposure: the mean luminance of the color input, then the exposure of this frame in the last block.
import Numerics;

[[vk::binding(0)]] Texture2D<float4> color;
[[vk::binding(1)]] [[vk::image_format("r16f")]] RWTexture2D<float> exposure;
[[vk::binding(2)]] Texture2D<float> previousExposure;
//! Bytes 0 to 7: the sum of the block sums in fixed point. Bytes 8 to 11: the count of finished blocks.
[[vk::binding(3)]] RWByteAddressBuffer current;
//! The buffer of the next frame (257 words), cleared by the last block.
[[vk::binding(4)]] RWByteAddressBuffer next;
//! float4: exposure, previous exposure, 1 / exposure, log2 of the exposure change / 6.
[[vk::binding(5)]] RWByteAddressBuffer result;

struct Constants {
    int2 offset;
    uint2 last;
    uint blockCount;
    float meanScale;
    uint reset;
};
[[vk::push_constant]] ConstantBuffer<Constants> constants;

groupshared float sums[256];
groupshared uint lastBlock;

[numthreads(16, 16, 1)]
[shader("compute")]
void histogramBasic(uint3 id : SV_DispatchThreadID, uint3 local : SV_GroupThreadID)
{
    uint thread = local.y * 16 + local.x;
    bool inside = id.x <= constants.last.x && id.y <= constants.last.y;
    sums[thread] = 0.0f;
    if (thread == 0)
        lastBlock = 0;
    float luminance = 0.0f;
    if (inside) {
        float4 texel = color.Load(int3(int2(id.xy) + constants.offset, 0));
        luminance = fma(texel.z, 0.25f, fma(texel.x, 0.25f, texel.y * 0.5f));
        luminance = min(max(luminance, 0.0f), 10000.0f);
    }
    GroupMemoryBarrierWithGroupSync();
    if (inside)
        sums[thread] = luminance;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (thread < stride)
            sums[thread] = sums[thread] + sums[thread + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread == 0) {
        uint64_t fixed = uint64_t(roundNearestEven(sums[0] * 429496.71875f));
        current.InterlockedAdd64(0, fixed);
        DeviceMemoryBarrier();
        uint finished;
        current.InterlockedAdd(8, 1u, finished);
        lastBlock = finished + 1 == constants.blockCount ? 1 : 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lastBlock == 0)
        return;
    if (thread == 0) {
        uint64_t total;
        current.InterlockedAdd64(0, 0, total);
        float previous = previousExposure.Load(int3(0, 0, 0));
        float mean = max(float(total) * constants.meanScale, 0.0049999998882412910461f);
        float value = rcp(mean) * 0.23547089099884033203f;
        exposure[uint2(0, 0)] = value;
        float stored = roundTowardZeroHalf(value);
        float4 values;
        values.x = stored != 0.0f ? stored : 1.0f;
        values.y = previous != 0.0f ? previous : 1.0f;
        values.z = rcp(values.x);
        values.w = 0.0f;
        if (constants.reset == 0) {
            float ratio = max(values.x, 9.9999999392252902908e-09f) * rcp(max(values.y, 9.9999999392252902908e-09f));
            values.w = min(max(log2(ratio), -6.0f), 6.0f) * 0.16666667163372039795f;
        }
        result.Store4(0, asuint(values));
    }
    for (uint index = thread; index <= 256; index += 256)
        next.Store(index * 4, 0u);
}
