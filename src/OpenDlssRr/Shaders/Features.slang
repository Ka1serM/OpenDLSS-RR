//! The 43 input features of enc0 at a pixel of the L0 grid (the render size rounded up to 32; outside the render size the grid reads the
//! image mirrored), with the 8-bit blue-noise dither of the network, for the stem of Block.slang. The owner of a pixel also writes the
//! motion images that the next frame and post read.
//!
//!   0 ... 2    colour, log encoded                  3 ... 14   colour history at the nearest, centre (twice: half and quarter size) and
//!   15 ... 20  diffuse and specular albedo                     farthest motion of the 3x3 neighbourhood
//!   21 ... 24  normal and roughness                 25         motion change at the nearest motion
//!   26 ... 29  feedback (image 13) at the nearest   30 ... 33  0 (optional inputs that this integration does not give)
//!   34         exposure change                      35 ... 37  colour history at the motion of the reflection
//!   38         motion change of the reflection      39 ... 42  feedback at the motion of the reflection
//! Motions are in history (post) pixels; the history images are the post output of the last frame at half and quarter size.

module Features;
import Network;

//! The bindings follow those of Block.slang.
[[vk::binding(13)]] Texture2D<float4> color;
[[vk::binding(14)]] Texture2D<float4> diffuseAlbedo;
[[vk::binding(15)]] Texture2D<float4> specularAlbedo;
[[vk::binding(16)]] Texture2D<float4> normals;
[[vk::binding(17)]] Texture2D<float> roughness;
[[vk::binding(18)]] Texture2D<float> depth;
[[vk::binding(19)]] Texture2D<float2> motionVectors;
[[vk::binding(20)]] Texture2D<float> specularHitDistance;
//! The motion images of the last frame (nearest motion xy, reflection motion zw), the history (log encoded, half and quarter size) and the
//! feedback image 13; all bilinear with clamp to edge.
[[vk::binding(21)]] Sampler2D<float4> previousMotion;
[[vk::binding(22)]] Sampler2D<float4> history;
[[vk::binding(23)]] Sampler2D<float4> pooledHistory;
[[vk::binding(24)]] Sampler2D<float4> previousFeedback;
//! x: exposure, y: previous exposure, w: exposure change (HistogramBasic).
[[vk::binding(25)]] StructuredBuffer<float4> exposure;
//! Render size: the nearest and the reflection motion (RGBA16F), the farthest and the centre motion (RG16F through R32_UINT views).
[[vk::binding(26)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> nearestAndReflectionMotion;
[[vk::binding(27)]] [[vk::image_format("r32ui")]] RWTexture2D<uint> farthestMotionBits;
[[vk::binding(28)]] [[vk::image_format("r32ui")]] RWTexture2D<uint> centreMotionBits;

struct Parameters {
    //! Camera matrices, row major for row vectors (v · M): the projection, render pixel to world (relative to the camera) and world to the
    //! pixel of the last frame.
    float4 viewToClip[4];
    float4 pixelToWorld[4];
    float4 worldToPreviousPixel[4];
    int2 size;
    int2 renderSize;
    float2 jitter;
    float2 previousJitter;
    //! Render size / history size and its inverse.
    float2 renderFromHistory;
    float2 historyFromRender;
    float2 inverseRenderSize;
    float2 inverseHistorySize;
    //! The history size, and the sizes of the half (image 11) and quarter (image 12) history images.
    float2 historySize;
    float2 halfSize;
    float2 inverseHalfSize;
    float2 inverseQuarterSize;
    //! The previous jitter / render size: the offset of the feedback image.
    float2 feedbackOffset;
    float motionDifferenceGain;
    float motionDifferenceScale;
    float historyScale;
    float feedbackScale;
    uint reset;
    //! The frame index modulo 8 selects the noise: the blue-noise thresholds (8 frames of 16x16, in units of the 7 FP16 bits that the
    //! dither drops) are in the FP32 weights at `noise`.
    uint frame;
    uint noise;
};
[[vk::binding(29)]] ConstantBuffer<Parameters> parameters;

float3 finiteOrZero(float3 value) {
    return float3(isfinite(value.x) ? value.x : 0.0f, isfinite(value.y) ? value.y : 0.0f, isfinite(value.z) ? value.z : 0.0f);
}

//! The colour encoding of the network: 0.36 · log10(1 + 100 x).
float3 logEncode(float3 value) {
    return 0.36f * log10(max(1.0f + 100.0f * value, 1e-4f));
}

//! The inverse of logEncode for the history images, which store code / 2.45.
float3 logDecode(float3 code) {
    code = clamp(code * 2.45f, 1e-4f, 3.0f);
    return (pow(10.0f, code / 0.36f) - 1.0f) / 100.0f;
}

//! A row vector (x, y, z, 1) times a 4x4 matrix.
float4 transform(float3 point, float4 matrix[4]) {
    return point.x * matrix[0] + point.y * matrix[1] + point.z * matrix[2] + matrix[3];
}

float3 safeNormalize(float3 v) {
    return v / (sqrt(dot(v, v)) + 1e-10f);
}

//! The motion of the reflection at a render pixel: the virtual image of the hit point lies on the view ray, the hit distance behind the
//! surface. Falls back to the surface motion where the depth or a projection is not valid.
float2 reflectionMotion(int2 pixel, float3 normal, float2 surfaceMotion) {
    float viewDepth = depth.Load(int3(pixel, 0));
    float4 m[4] = parameters.viewToClip;
    float ndc = m[3].w < 1.0f ? (viewDepth * m[2].z + m[3].z) / (viewDepth * m[2].w) : viewDepth * m[2].z + m[3].z;
    float distance = min(specularHitDistance.Load(int3(pixel, 0)), 1e7f);
    distance = distance <= 0.0f ? 1e7f : distance;
    float2 centre = float2(pixel) + 0.5f;
    float4 world = transform(float3(centre, ndc), parameters.pixelToWorld);
    float3 position = world.xyz / world.w;
    float3 n = safeNormalize(normal);
    float3 view = safeNormalize(position);
    float3 hit = distance * (view - 2.0f * dot(n, view) * n);
    float alongNormal = abs(dot(n, hit));
    float length = sqrt(max(dot(hit, hit) - alongNormal * alongNormal, 0.0f) + pow(max(alongNormal, 1e-5f), 2.0f));
    float4 previous = transform(position + view * length, parameters.worldToPreviousPixel);
    float2 motion = (previous.xy / previous.w - centre) * parameters.historyFromRender;
    bool valid = isfinite(viewDepth) && ndc != 1.0f && world.w > 0.0f && isfinite(world.w) && previous.w > 0.0f && isfinite(previous.w)
                 && all(isfinite(motion));
    return valid ? motion : surfaceMotion;
}

//! The change of a motion against the motion of the last frame at the reprojected pixel: 0.131 · ln(1 + 5.38 |Δ|), in render pixels.
float motionChange(float2 motion, float2 previous) {
    float2 difference = clamp(motion * parameters.renderFromHistory - previous * parameters.renderFromHistory, -1000.0f, 1000.0f);
    float length = sqrt(dot(difference * parameters.motionDifferenceGain, difference * parameters.motionDifferenceGain));
    return min(parameters.motionDifferenceScale * log(1.0f + length), 1000.0f);
}

//! The position in the history (post pixels) that a render pixel centre moves to, and whether the history has it.
struct Reprojection {
    float2 position;
    bool inside;
    __init(float2 centre, float2 motion) {
        position = motion + (centre - parameters.jitter) * parameters.historyFromRender;
        inside = all(position >= 0.0f) && all(position <= parameters.historySize) && parameters.reset == 0;
    }
}

//! The colour history of the last frame (log encoded, half or quarter size) at the reprojected pixel, in the exposure of this frame;
//! outside the history the colour of this frame.
float3 historyFeature(float2 centre, float2 motion, bool quarter, float3 colour, float4 exposures) {
    Reprojection reprojection = Reprojection(centre, motion);
    float2 uv = reprojection.position * parameters.inverseHalfSize;
    float4 code = quarter ? pooledHistory.SampleLevel(uv * parameters.halfSize * parameters.inverseQuarterSize, 0) : history.SampleLevel(uv, 0);
    float3 value = reprojection.inside ? logDecode(code.rgb) / exposures.y : colour;
    return min(finiteOrZero(logEncode(value * exposures.x) * parameters.historyScale), 1024.0f);
}

//! The feedback image 13 of the last frame at the reprojected pixel; outside the history 0.
float4 feedbackFeature(float2 centre, float2 motion) {
    Reprojection reprojection = Reprojection(centre, motion);
    float2 uv = reprojection.position * parameters.inverseHistorySize + parameters.feedbackOffset;
    return reprojection.inside ? previousFeedback.SampleLevel(uv, 0) * parameters.feedbackScale : float4(0.0f);
}

//! The 8-bit quantization of a feature: FP16, then the blue-noise stochastic rounding to 3 mantissa bits. noise in [0, 128) is the
//! threshold in units of the 7 dropped bits: the value is x + noise · ulp, truncated. The additions round to FP16 as in the network.
float dither(float value, uint noise) {
    uint bits = halfBits(value);
    float power = f16tof32(bits & 0xFC00u);
    float nudged = f16tof32(halfBits(f16tof32(bits) + f16tof32((bits & 0xFC00u) | noise)));
    return f16tof32(halfBits(nudged - power) & 0xFF80u);
}

//! The L0 grid size, the frame and the offset of its blue noise.
public int2 gridSize() {
    return parameters.size;
}

public uint noiseAt(int2 grid) {
    return parameters.noise + parameters.frame * 256 + uint((grid.y & 15) * 16 + (grid.x & 15));
}

void put<let N : int>(inout float values[24], int at, vector<float, N> value) {
    for (int c = 0; c < N; ++c)
        values[at + c] = value[c];
}

//! Half of the 48 features (43 and 5 zeros) of a grid pixel, dithered with the threshold `noise`: part 0 the channels 0 ... 23 and the
//! exposed colour, part 1 the channels 24 ... 47. The owner writes the motion images.
public void features(int2 grid, int part, uint noise, bool owner, out float values[24], out float3 exposed)
{
    int2 render = parameters.renderSize;
    int2 pixel = int2(grid.x < render.x ? grid.x : 2 * render.x - 2 - grid.x, grid.y < render.y ? grid.y : 2 * render.y - 2 - grid.y);
    float4 exposures = exposure[0];
    for (int c = 0; c < 24; ++c)
        values[c] = 0.0f;
    exposed = float3(0.0f);

    //! The 3x3 neighbours with the smallest and the largest |depth| (the first one wins a tie; the centre comes first).
    int2 nearest = pixel;
    int2 farthest = pixel;
    float nearestDepth = abs(depth.Load(int3(pixel, 0)));
    float farthestDepth = nearestDepth;
    for (int index = 0; index < 9; ++index) {
        int2 offset = int2(index % 3 - 1, index / 3 - 1);
        if (all(offset == 0))
            continue;
        int2 neighbour = clamp(pixel + offset, int2(0), render - 1);
        float value = abs(depth.Load(int3(neighbour, 0)));
        if (value < nearestDepth) {
            nearest = neighbour;
            nearestDepth = value;
        } else if (value > farthestDepth) {
            farthest = neighbour;
            farthestDepth = value;
        }
    }
    float2 nearestMotion = motionVectors.Load(int3(nearest, 0)) * parameters.historyFromRender;
    float2 surfaceMotion = motionVectors.Load(int3(pixel, 0)) * parameters.historyFromRender;
    float2 centre = float2(pixel) + 0.5f;
    float3 raw = color.Load(int3(pixel, 0)).rgb;
    float4 normal = normals.Load(int3(pixel, 0));
    float3 diffuse = diffuseAlbedo.Load(int3(pixel, 0)).rgb;
    float3 specular = specularAlbedo.Load(int3(pixel, 0)).rgb;
    bool lit = diffuse.x + diffuse.y + diffuse.z + (specular.x + specular.y + specular.z) != 0.0f;

    if (part == 0) {
        float2 farthestMotion = motionVectors.Load(int3(farthest, 0)) * parameters.historyFromRender;
        exposed = min(finiteOrZero(raw * exposures.x), 65504.0f);
        put(values, 0, logEncode(exposed));
        put(values, 3, historyFeature(centre, nearestMotion, false, raw, exposures));
        put(values, 6, historyFeature(centre, surfaceMotion, false, raw, exposures));
        put(values, 9, historyFeature(centre, surfaceMotion, true, raw, exposures));
        put(values, 12, historyFeature(centre, farthestMotion, false, raw, exposures));
        put(values, 15, min(diffuse, 2.0f));
        put(values, 18, min(specular, 2.0f));
        put(values, 21, lit ? min(normal.xyz, 2.0f) : float3(0.0f));
        for (int c = 0; c < 24; ++c)
            values[c] = dither(values[c], noise);
        if (owner && all(grid < render))
            farthestMotionBits[grid] = halfBits(farthestMotion.x) | halfBits(farthestMotion.y) << 16;
        return;
    }

    //! The motion of the last frame at the reprojected pixel.
    float2 glossyMotion = reflectionMotion(pixel, normal.xyz, surfaceMotion);
    float2 base = centre - parameters.jitter + parameters.previousJitter;
    float2 uv = (base + nearestMotion * parameters.renderFromHistory) * parameters.inverseRenderSize;
    float nearestChange = motionChange(nearestMotion, previousMotion.SampleLevel(uv, 0).xy);
    uv = (base + glossyMotion * parameters.renderFromHistory) * parameters.inverseRenderSize;
    float glossyChange = motionChange(glossyMotion, previousMotion.SampleLevel(uv, 0).zw);
    float keep = parameters.reset != 0 ? 0.0f : 1.0f;
    values[0] = lit ? min(roughness.Load(int3(pixel, 0)), 2.0f) : 0.0f;
    values[1] = nearestChange * keep;
    put(values, 2, feedbackFeature(centre, nearestMotion));
    values[10] = exposures.w;
    put(values, 11, historyFeature(centre, glossyMotion, false, raw, exposures));
    values[14] = glossyChange * keep;
    put(values, 15, feedbackFeature(centre, glossyMotion));
    for (int c = 0; c < 24; ++c)
        values[c] = dither(values[c], noise);
    if (!owner || any(grid >= render))
        return;
    nearestAndReflectionMotion[grid] = float4(nearestMotion, glossyMotion);
    centreMotionBits[grid] = halfBits(surfaceMotion.x) | halfBits(surfaceMotion.y) << 16;
}
