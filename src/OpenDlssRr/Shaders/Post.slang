//! Post: the network output (render size, linear) to the 3x output size with an anisotropic 4x4 Gaussian filter in FP16,
//! then a blend with the reprojected history (four motion candidates, bicubic history samples) in linear light.
//! Writes the new history (log domain, FP16), the input of the downsample kernel (log domain, FP16 arrays) and a half-size mip (unorm10).
import Numerics;
import BicubicPhaseWeights;

//! The source texels (FP16 red, green, blue, unused) and the two FP16 output arrays of the downsample kernel.
[[vk::binding(0)]] RWByteAddressBuffer data;
//! float4: exposure, previous exposure, 1 / exposure, exposure change.
[[vk::binding(1)]] RWByteAddressBuffer exposure;
//! Render size, unorm8: x filtered as blend weight, y and z filtered as the two blend factors.
[[vk::binding(2)]] Texture2D<float4> guide;
//! Render size, unorm8, bilinear: the weights of the four motion candidates.
[[vk::binding(3)]] Sampler2D<float4> candidateWeights;
//! Render size, FP16: the Gaussian shape (x², y², xy coefficients).
[[vk::binding(4)]] Texture2D<float4> shape;
//! Output size, FP16, bilinear: the history in linear light.
[[vk::binding(5)]] Sampler2D<float4> history;
//! Render size, FP16: the motion of candidate 1 (xy) and of candidate 4 (zw), in output pixels.
[[vk::binding(6)]] Texture2D<float4> motionA;
//! Render size, FP16: the motion of candidate 3.
[[vk::binding(7)]] Texture2D<float2> motionC;
//! Render size, FP16, bilinear: the motion of candidate 2.
[[vk::binding(8)]] Sampler2D<float2> motionB;
//! Output size, RGBA16F through an R32G32_UINT view: the new history in the log domain.
[[vk::binding(9)]] [[vk::image_format("rg32ui")]] RWTexture2D<uint2> target;
//! Half output size, A2B10G10R10 unorm through an R32_UINT view.
[[vk::binding(10)]] [[vk::image_format("r32ui")]] RWTexture2D<uint> mip;

struct Constants {
    float2 scale;
    float2 offset;
    int2 outputSize;
    float2 historyInverseSize;
    int2 inputSize;
    int2 origin;
    int2 end;
    int2 pitch;
    float2 inputInverseSize;
    float blendScale;
    int sourcePitch;
    uint sourceOffset;
    uint colorOffset;
    uint blueOffset;
    uint reset;
};
[[vk::push_constant]] ConstantBuffer<Constants> constants;

//! A group writes 16x32 post pixels and reads the input texels under them, with two texels of margin on each side. The render size is at most
//! the output size and the post size is 1.5 times the output size, so the scale is at most 2/3: 16·2/3 + 5 columns and 32·2/3 + 6 rows.
static const uint tileWidth = 16;
static const uint tileHeight = 28;
groupshared half4 colorTile[tileWidth * tileHeight];
groupshared half4 guideTile[tileWidth * tileHeight];
groupshared half2 shapeTile[tileWidth * tileHeight];
//! The colour sum (xyz) and the count (w) of the two rows of each thread, for the mip.
groupshared float4 rowSums[256];

//! log10(100 x + 1) * 0.36, the log domain of the network.
float toLog(float value) {
    float encoded = log2(max(fma(value, 100.0f, 1.0f), 9.9999997473787516356e-05f));
    encoded = encoded * 0.30103000998497009277f;
    return encoded * 0.36000001430511474609f;
}

//! 100 times the inverse of toLog, for codes clamped to [1e-4, 3].
float toLinearScaled(float code) {
    float clamped = code > 9.9999997473787516356e-05f || isnan(code) ? minNan(code, 3.0f) : 9.9999997473787516356e-05f;
    float exponent = clamped * 2.7777776718139648438f;
    exponent = exponent * 3.3219280242919921875f;
    return exp2(exponent) - 1.0f;
}

//! The inverse of toLog for codes clamped to [1e-4, 3].
float toLinear(float code) {
    return toLinearScaled(code) * 0.0099999997764825820923f;
}

//! A texel at integer coordinates, 0 outside the image.
float4 fetch(Texture2D<float4> texture, int2 position) {
    if (any(position < 0) || any(position >= constants.inputSize))
        return 0.0f;
    return texture.Load(int3(position, 0));
}

float2 fetch(Texture2D<float2> texture, int2 position) {
    if (any(position < 0) || any(position >= constants.inputSize))
        return 0.0f;
    return texture.Load(int3(position, 0));
}

struct Candidate {
    float weight;
    float2 motion;
};

//! Moves the candidate with the larger weight first. NaN weights do not move.
void order(inout Candidate first, inout Candidate second) {
    if (first.weight < second.weight) {
        Candidate larger = second;
        second = first;
        first = larger;
    }
}

//! The history at an output position in pixels: a 5-tap bicubic filter from bilinear samples.
float3 sampleHistory(float2 position) {
    float2 center = floor(position - 0.5f) + 0.5f;
    float2 phase = position - center;
    uint2 index = uint2(fma(saturate(phase), 512.0f, 0.5f));
    index = min(index, 512u);
    float4 weightsX = asfloat(bicubicPhaseWeights[index.x]);
    float4 weightsY = asfloat(bicubicPhaseWeights[index.y]);
    float2 low = (center - 1.0f) * constants.historyInverseSize;
    float2 middle = float2(center.x + weightsX.w, center.y + weightsY.w) * constants.historyInverseSize;
    float2 high = (center + 2.0f) * constants.historyInverseSize;
    float3 left = history.SampleLevel(float2(low.x, middle.y), 0.0f).xyz;
    float3 top = history.SampleLevel(float2(middle.x, low.y), 0.0f).xyz;
    float3 centerSample = history.SampleLevel(middle, 0.0f).xyz;
    float3 bottom = history.SampleLevel(float2(middle.x, high.y), 0.0f).xyz;
    float3 right = history.SampleLevel(float2(high.x, middle.y), 0.0f).xyz;
    float weightLeft = weightsX.x * weightsY.y;
    float weightTop = weightsX.y * weightsY.x;
    float weightCenter = weightsX.y * weightsY.y;
    float weightBottom = weightsX.y * weightsY.z;
    float weightRight = weightsX.z * weightsY.y;
    float total = weightLeft + weightTop;
    total = weightCenter + total;
    total = weightBottom + total;
    total = weightRight + total;
    float3 color = weightTop * top;
    color = fma(left, weightLeft, color);
    color = fma(weightCenter, centerSample, color);
    color = fma(weightBottom, bottom, color);
    color = fma(weightRight, right, color);
    return rcp(total) * color;
}

//! The history of one candidate in the log domain, or the fallback if the motion leaves the image.
float3 candidateHistory(Candidate candidate, float2 pixelCenter, float3 fallback) {
    float2 position = float2(pixelCenter.x + candidate.motion.x, candidate.motion.y + pixelCenter.y);
    if (position.x < 0.0f || position.x > float(constants.outputSize.x) || position.y < 0.0f || position.y > float(constants.outputSize.y)
        || constants.reset != 0)
        return fallback;
    return sampleHistory(position);
}

//! The filter result of one output row.
struct Filtered {
    float weightSum;
    float3 color;
    float3 guide;
};

struct Row {
    bool written;
    float3 blended;
    uint2 packedTarget;
    uint colorBits;
    uint blueBits;
};

Row resolveRow(int x, int y, float2 pixelCenter, float sourceX, float sourceY, float sampleY, Filtered filtered, float4 exposures)
{
    Row row;
    row.written = false;
    row.blended = 0.0f;
    if (x >= constants.end.x || y >= constants.end.y || x >= constants.outputSize.x || y >= constants.outputSize.y)
        return row;
    row.written = true;
    float sourceU = sourceX * constants.inputInverseSize.x;
    float4 keys = candidateWeights.SampleLevel(float2(sourceU, sampleY), 0.0f);
    float normalize = rcp(max(filtered.weightSum, 1.00000001335143196e-10f));
    float3 currentScaled;
    [ForceUnroll]
    for (int channel = 0; channel < 3; ++channel)
        currentScaled[channel] = toLinearScaled(normalize * filtered.color[channel]);
    float3 current = currentScaled * 0.0099999997764825820923f;
    int2 texel = int2(int(floor(sourceX)), int(floor(sourceY)));
    float4 motion14 = fetch(motionA, texel);
    float2 motion3 = fetch(motionC, texel);
    float2 motion2 = motionB.SampleLevel(float2(sourceU, sourceY * constants.inputInverseSize.y), 0.0f);
    Candidate a = { keys.x, motion14.xy };
    Candidate b = { keys.y, motion2 };
    Candidate c = { keys.z, motion3 };
    Candidate d = { keys.w, motion14.zw };
    order(a, b);
    order(c, d);
    order(a, c);
    order(b, d);
    order(b, c);
    float weightA = saturate(a.weight);
    float weightB = saturate(b.weight);
    float weightC = saturate(c.weight);
    float weightD = saturate(d.weight);
    float total = weightA + weightB;
    total = weightC + total;
    total = weightD + total;
    float inverse = rcp(max(total, 1.00000001335143196e-10f));
    weightA = weightA * inverse;
    weightB = weightB * inverse;
    weightC = weightC * inverse;
    weightD = weightD * inverse;

    float3 fallback;
    [ForceUnroll]
    for (int channel = 0; channel < 3; ++channel)
        fallback[channel] = toLog(exposures.y * (exposures.z * current[channel]));
    float3 accumulated = weightA * candidateHistory(a, pixelCenter, fallback);
    float weight = weightA;
    if (weight < 0.89999997615814208984f) {
        accumulated = fma(weightB, candidateHistory(b, pixelCenter, fallback), accumulated);
        weight = weight + weightB;
        if (weight < 0.89999997615814208984f) {
            accumulated = fma(weightC, candidateHistory(c, pixelCenter, fallback), accumulated);
            weight = weight + weightC;
            if (weight < 0.89999997615814208984f) {
                accumulated = fma(weightD, candidateHistory(d, pixelCenter, fallback), accumulated);
                weight = weightD + weight;
            }
        }
    }
    float inverseWeight = rcp(weight);
    float3 previous;
    [ForceUnroll]
    for (int channel = 0; channel < 3; ++channel)
        previous[channel] = exposures.x * (toLinear(accumulated[channel] * inverseWeight) / exposures.y);

    float factor = saturate((constants.blendScale * filtered.weightSum) * (normalize * filtered.guide.x));
    float factorTarget = fma(factor, 1.0f - normalize * filtered.guide.y, normalize * filtered.guide.y);
    float factorOutput = fma(factor, 1.0f - normalize * filtered.guide.z, normalize * filtered.guide.z);
    //! The scale of the current color is fused into the subtraction.
    float3 difference = fma(currentScaled, 0.0099999997764825820923f, -previous);
    float3 blendedTarget = fma(difference, factorTarget, previous);
    if (isnan((blendedTarget.x + blendedTarget.y) + blendedTarget.z))
        blendedTarget = current;
    float3 blendedOutput = fma(difference, factorOutput, previous);
    if (isnan((blendedOutput.x + blendedOutput.y) + blendedOutput.z))
        blendedOutput = current;

    uint3 targetBits;
    uint3 outputBits;
    [ForceUnroll]
    for (int channel = 0; channel < 3; ++channel) {
        targetBits[channel] = uint(bit_cast<uint16_t>(half(toLog(blendedTarget[channel]))));
        outputBits[channel] = uint(bit_cast<uint16_t>(half(toLog(blendedOutput[channel]))));
    }
    row.blended = blendedTarget;
    row.packedTarget = uint2(targetBits.x | targetBits.y << 16, targetBits.z);
    row.colorBits = outputBits.x | outputBits.y << 16;
    row.blueBits = outputBits.z;
    return row;
}

void storeOutput(Row row, int localX, int localY)
{
    if (!row.written || localX >= constants.pitch.x || (localX | localY) < 0 || localY >= constants.pitch.y)
        return;
    uint element = uint(constants.pitch.x * localY + localX);
    data.Store(constants.colorOffset + element * 4, row.colorBits);
    data.Store<uint16_t>(constants.blueOffset + element * 2, uint16_t(row.blueBits));
}

[numthreads(16, 16, 1)]
[shader("compute")]
void post(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID)
{
    uint thread = local.y * 16 + local.x;
    int blockX = constants.origin.x + int(group.x * 16);
    int blockY = constants.origin.y + int(group.y * 32);
    int tileX = int(constants.scale.x * float(blockX));
    int tileY = int(constants.scale.y * float(blockY));
    float4 exposures = asfloat(exposure.Load4(0));

    for (uint texel = thread; texel < tileWidth * tileHeight; texel += 256) {
        int2 position = int2(tileX + int(texel % tileWidth) - 2, tileY + int(texel / tileWidth) - 2);
        position.x = position.x > 0 ? min(position.x, constants.inputSize.x - 1) : 0;
        position.y = position.y > 0 ? min(position.y, constants.inputSize.y - 1) : 0;
        uint2 packed = data.Load2(constants.sourceOffset + uint(constants.sourcePitch * position.y + position.x) * 8);
        half3 source = half3(bit_cast<half2>(packed.x), bit_cast<half2>(packed.y).x);
        half4 guideTexel = half4(guide.Load(int3(position, 0)));
        float4 shapeTexel = shape.Load(int3(position, 0));
        colorTile[texel] = half4(half(toLog(float(source.x))), half(toLog(float(source.y))), half(toLog(float(source.z))), guideTexel.x);
        guideTile[texel] = half4(guideTexel.y, guideTexel.z, 0.0h, half(shapeTexel.x * -0.72132498025894165039f));
        shapeTile[texel] = half2(half(shapeTexel.y * -0.72132498025894165039f), half(shapeTexel.z * -1.4426499605178833008f));
    }
    GroupMemoryBarrierWithGroupSync();

    int x = constants.origin.x + int(group.x * 16 + local.x);
    int y = constants.origin.y + 2 * int(group.y * 16 + local.y);
    float centerX = float(x) + 0.5f;
    float sourceX = fma(centerX, constants.scale.x, constants.offset.x);
    float centerY0 = float(y) + 0.5f;
    float sourceY0 = fma(constants.scale.y, centerY0, constants.offset.y);
    float centerY1 = float(y + 1) + 0.5f;
    float sourceY1 = fma(constants.scale.y, centerY1, constants.offset.y);
    int baseX = int(floor(sourceX - 0.5f));
    int baseY0 = int(floor(sourceY0 - 0.5f));
    int baseY1 = int(floor(sourceY1 - 0.5f));
    half offsetX = half((float(baseX) + 0.5f) - sourceX);
    half offsetY0 = half((float(baseY0) + 0.5f) - sourceY0);
    half offsetY1 = half((float(baseY1) + 0.5f) - sourceY1);
    if (baseY0 != baseY1)
        offsetY1 = offsetY1 - 1.0h;
    half2 offsetY = half2(offsetY0, offsetY1);

    //! Both rows use the taps around the first row; the lanes of each half2 are the two rows.
    half2 accumulators[6];
    [ForceUnroll]
    for (int channel = 0; channel < 6; ++channel)
        accumulators[channel] = half2(0.0h, 0.0h);
    float2 weightSum = 0.0f;
    [ForceUnroll]
    for (int dx = -1; dx <= 2; ++dx) {
        half tapX = offsetX + half(dx);
        half tapX2 = tapX * tapX;
        [ForceUnroll]
        for (int dy = -1; dy <= 2; ++dy) {
            uint index = uint(baseY0 - tileY + 2 + dy) * tileWidth + uint(baseX - tileX + 2 + dx);
            half4 color = colorTile[index];
            half4 guideTexel = guideTile[index];
            half2 shapeTexel = shapeTile[index];
            half2 tapY = offsetY + half2(half(dy), half(dy));
            half2 tapY2 = tapY * tapY;
            half termX = tapX2 * guideTexel.w;
            half2 exponent = fma(tapY2, half2(shapeTexel.x, shapeTexel.x), half2(termX, termX));
            half2 tapXY = half2(tapX, tapX) * tapY;
            exponent = fma(tapXY, half2(shapeTexel.y, shapeTexel.y), exponent);
            float2 weight = float2(exp2(float(exponent.x)), exp2(float(exponent.y)));
            half2 weightHalf = half2(half(weight.x), half(weight.y));
            accumulators[0] = fma(weightHalf, half2(color.x, color.x), accumulators[0]);
            accumulators[1] = fma(weightHalf, half2(color.y, color.y), accumulators[1]);
            accumulators[2] = fma(weightHalf, half2(color.z, color.z), accumulators[2]);
            accumulators[3] = fma(weightHalf, half2(color.w, color.w), accumulators[3]);
            accumulators[4] = fma(weightHalf, half2(guideTexel.x, guideTexel.x), accumulators[4]);
            accumulators[5] = fma(weightHalf, half2(guideTexel.y, guideTexel.y), accumulators[5]);
            weightSum = weightSum + weight;
        }
    }

    Filtered filtered0;
    filtered0.weightSum = weightSum.x;
    filtered0.color = float3(float(accumulators[0].x), float(accumulators[1].x), float(accumulators[2].x));
    filtered0.guide = float3(float(accumulators[3].x), float(accumulators[4].x), float(accumulators[5].x));
    Filtered filtered1;
    filtered1.weightSum = weightSum.y;
    filtered1.color = float3(float(accumulators[0].y), float(accumulators[1].y), float(accumulators[2].y));
    filtered1.guide = float3(float(accumulators[3].y), float(accumulators[4].y), float(accumulators[5].y));

    Row row0 = resolveRow(x, y, float2(centerX, centerY0), sourceX, sourceY0, sourceY0 * constants.inputInverseSize.y, filtered0, exposures);
    Row row1 = resolveRow(x, y + 1, float2(centerX, centerY1), sourceX, sourceY1, sourceY1 * constants.inputInverseSize.y, filtered1, exposures);
    if (row0.written)
        target[uint2(x, y)] = row0.packedTarget;
    if (row1.written)
        target[uint2(x, y + 1)] = row1.packedTarget;

    int localX = int(group.x * 16 + local.x);
    int localY = 2 * int(group.y * 16 + local.y);
    storeOutput(row0, localX, localY);
    storeOutput(row1, localX, localY + 1);

    float3 sum = 0.0f;
    uint count = 0;
    if (row0.written) {
        sum = row0.blended + 0.0f;
        count = 1;
    }
    if (row1.written) {
        sum = sum + row1.blended;
        count = count + 1;
    }
    //! A mip pixel is the mean of the 2x2 output pixels of a thread pair (columns x and x + 1).
    rowSums[thread] = float4(sum, float(count));
    GroupMemoryBarrierWithGroupSync();
    float4 neighbor = rowSums[thread ^ 1u];
    float3 total = sum + neighbor.xyz;
    int totalCount = int(neighbor.w) + int(count);
    if ((x & 1) != 0)
        return;
    int2 mipPosition = int2(x / 2, y / 2);
    if (mipPosition.x >= (constants.outputSize.x + 1) / 2 || totalCount < 1 || mipPosition.y >= (constants.outputSize.y + 1) / 2)
        return;
    float scale = rcp(float(totalCount));
    float3 value;
    [ForceUnroll]
    for (int channel = 0; channel < 3; ++channel)
        value[channel] = toLog(total[channel] * scale) * 0.40816324949264526367f;
    mip[uint2(mipPosition)] = unorm10(value.x) | unorm10(value.y) << 10 | unorm10(value.z) << 20 | 3u << 30;
}
