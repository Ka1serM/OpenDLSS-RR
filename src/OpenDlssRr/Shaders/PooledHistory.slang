//! Pooled history: 2x2 average of the history in a linear light domain, then back to the log domain, at half size.
import Numerics;

[[vk::binding(0)]] Texture2D<float4> source;
[[vk::binding(1)]] [[vk::image_format("r32ui")]] RWTexture2D<uint> target;

struct Constants {
    int2 sourceSize;
    int2 targetSize;
};
[[vk::push_constant]] ConstantBuffer<Constants> constants;

float toLinear(float code) {
    precise float scaled = code * 2.4500000476837158203f;
    precise float clamped = scaled > 9.9999997473787516356e-05f || isnan(scaled) ? minNan(scaled, 3.0f) : 9.9999997473787516356e-05f;
    precise float exponent = clamped * 2.7777776718139648438f;
    exponent = exponent * 3.3219280242919921875f;
    precise float linear = exp2(exponent) - 1.0f;
    return linear;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void pooledHistory(uint3 id : SV_DispatchThreadID)
{
    int2 pixel = int2(id.xy);
    if (pixel.x >= constants.targetSize.x || pixel.y >= constants.targetSize.y)
        return;
    precise float3 sum = 0.0f;
    int count = 0;
    for (int dy = 0; dy < 2; ++dy) {
        int y = pixel.y * 2 + dy;
        if (y >= constants.sourceSize.y)
            break;
        for (int dx = 0; dx < 2; ++dx) {
            int x = pixel.x * 2 + dx;
            if (x >= constants.sourceSize.x)
                break;
            float4 texel = source.Load(int3(x, y, 0));
            sum.x = fma(toLinear(texel.x), 0.0099999997764825820923f, sum.x);
            sum.y = fma(toLinear(texel.y), 0.0099999997764825820923f, sum.y);
            sum.z = fma(toLinear(texel.z), 0.0099999997764825820923f, sum.z);
            count++;
        }
    }
    if (count == 0)
        return;
    precise float reciprocal = rcp(float(count));
    precise float3 result;
    [ForceUnroll]
    for (int channel = 0; channel < 3; ++channel) {
        precise float value = max(fma(reciprocal * sum[channel], 100.0f, 1.0f), 9.9999997473787516356e-05f);
        precise float logarithm = log2(value) * 0.30103000998497009277f;
        logarithm = logarithm * 0.36000001430511474609f;
        result[channel] = logarithm * 0.40816324949264526367f;
    }
    target[pixel] = unorm10(result.x) | unorm10(result.y) << 10 | unorm10(result.z) << 20 | 3u << 30;
}
