//! Downsample: the network output (log domain, FP16) at 1.5x size to the output size with a separable 5-tap filter
//! in FP16, clamped to the minimum and maximum of the taps, then to linear light divided by the exposure.
import Numerics;

//! Per source texel: a half2 (red, green) in the color array and a half (blue) in the blue array.
[[vk::binding(0)]] RWByteAddressBuffer data;
[[vk::binding(1)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> output;
[[vk::binding(2)]] RWByteAddressBuffer exposure;

struct Constants {
    int2 sourceSize;
    int2 outputSize;
    int2 outputOffset;
    int2 origin;
    int2 end;
    int2 pitch;
    int2 regionBegin;
    int2 regionEnd;
    uint colorOffset;
    uint blueOffset;
};
[[vk::push_constant]] ConstantBuffer<Constants> constants;

static const uint tileSize = 28;
groupshared uint2 tile[tileSize * tileSize];
groupshared uint2 rows[tileSize * 16];

struct Texel {
    half2 redGreen;
    half blue;
};

Texel unpack(uint2 packed) {
    Texel texel;
    texel.redGreen = bit_cast<half2>(packed.x);
    texel.blue = bit_cast<half>(uint16_t(packed.y & 0xffffu));
    return texel;
}

uint2 pack(Texel texel) {
    return uint2(bit_cast<uint>(texel.redGreen), uint(bit_cast<uint16_t>(texel.blue)));
}

//! The tap weights of the 1.5x filter as FP16 bits (0.038284853, 0.041347627, 0.37212864, 0.087289415 and 0.6201172, rounded toward zero).
//! Odd output coordinates use the mirrored weights.
half weight(int tap, bool odd) {
    if (tap == 0)
        return bit_cast<half>(uint16_t(0x38f6));
    bool near = (tap < 0) == odd;
    if (tap == -2 || tap == 2)
        return bit_cast<half>(uint16_t(near ? 0xa8e7 : 0xa94b));
    return bit_cast<half>(uint16_t(near ? 0x35f4 : 0x2d96));
}

//! slangc does not run default member initializers for `Filter filter;`, so the constructor sets each member.
struct Filter {
    half2 redGreen;
    half blue;
    half2 lowRedGreen;
    half lowBlue;
    half2 highRedGreen;
    half highBlue;

    __init() {
        redGreen = half2(0.0h, 0.0h);
        blue = 0.0h;
        lowRedGreen = half2(65504.0h, 65504.0h);
        lowBlue = 65504.0h;
        highRedGreen = half2(-65504.0h, -65504.0h);
        highBlue = -65504.0h;
    }

    [mutating]
    void add(Texel texel, half tapWeight) {
        redGreen = fma(half2(tapWeight, tapWeight), texel.redGreen, redGreen);
        blue = fma(tapWeight, texel.blue, blue);
        lowRedGreen = min(lowRedGreen, texel.redGreen);
        lowBlue = min(lowBlue, texel.blue);
        highRedGreen = max(highRedGreen, texel.redGreen);
        highBlue = max(highBlue, texel.blue);
    }

    Texel result() {
        Texel texel;
        texel.redGreen = min(max(redGreen, lowRedGreen), highRedGreen);
        texel.blue = min(max(blue, lowBlue), highBlue);
        return texel;
    }
};

//! 2^(code * 9.21) for the log-domain code of the network, before the offset of 1 is removed.
float toLinear(float code) {
    float clamped = code > 9.9999997473787516356e-05f || isnan(code) ? minNan(code, 3.0f) : 9.9999997473787516356e-05f;
    float exponent = clamped * 2.7777776718139648438f;
    exponent = exponent * 3.3219280242919921875f;
    return exp2(exponent);
}

[numthreads(16, 16, 1)]
[shader("compute")]
void downsample(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint thread = local.y * 16 + local.x;
    int2 base = int2(group.xy * 16) + constants.origin;
    int2 first = int2(floor(float2(uint2(base)) * 1.5f)) - 2;

    for (uint index = thread; index < tileSize * tileSize; index += 256) {
        int2 position = first + int2(index % tileSize, index / tileSize);
        position.x = position.x > 0 ? min(position.x, constants.sourceSize.x - 1) : 0;
        position.y = position.y > 0 ? min(position.y, constants.sourceSize.y - 1) : 0;
        int2 inRegion = position - constants.regionBegin;
        bool valid = all(position >= constants.regionBegin) && all(position < constants.regionEnd) && (inRegion.x | inRegion.y) > -1
                     && all(inRegion < constants.pitch);
        uint2 packed = 0;
        if (valid) {
            uint element = uint(inRegion.y * constants.pitch.x + inRegion.x);
            packed.x = data.Load(constants.colorOffset + element * 4);
            packed.y = uint(data.Load<uint16_t>(constants.blueOffset + element * 2));
        }
        tile[index] = packed;
    }
    GroupMemoryBarrierWithGroupSync();

    int outputX = base.x + int(local.x);
    bool oddX = ((int(local.x) + constants.origin.x) & 1) == 1;
    int centerX = int(floor((float(outputX) + 0.5f) * 1.5f)) - first.x;
    for (uint row = local.y; row < tileSize; row += 16) {
        Filter filter = Filter();
        for (int tap = -2; tap <= 2; ++tap) {
            int column = centerX + tap;
            if (uint(column) < tileSize)
                filter.add(unpack(tile[row * tileSize + column]), weight(tap, oddX));
        }
        rows[row * 16 + local.x] = pack(filter.result());
    }
    GroupMemoryBarrierWithGroupSync();

    int outputY = base.y + int(local.y);
    if (outputY >= constants.end.y || outputX >= constants.end.x || outputX >= constants.outputSize.x || outputY >= constants.outputSize.y)
        return;
    bool oddY = (outputY & 1) == 1;
    int centerY = int(floor((float(outputY) + 0.5f) * 1.5f)) - first.y;
    Filter filter = Filter();
    for (int tap = -2; tap <= 2; ++tap)
        filter.add(unpack(rows[(centerY + tap) * 16 + local.x]), weight(tap, oddY));
    Texel texel = filter.result();
    float scale = rcp(asfloat(exposure.Load(0)));
    float3 code = float3(float(texel.redGreen.x), float(texel.redGreen.y), float(texel.blue));
    float4 color = 1.0f;
    [ForceUnroll]
    for (int channel = 0; channel < 3; ++channel) {
        float linear = (toLinear(code[channel]) - 1.0f) * 0.0099999997764825820923f;
        color[channel] = linear * scale;
    }
    output[uint2(int2(outputX, outputY) + constants.outputOffset)] = color;
}
