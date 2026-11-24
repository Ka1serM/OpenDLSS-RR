//! Rounding helpers with an exactly defined result on all GPUs.
module Numerics;

//! The minimum, or NaN if an operand is NaN.
public float minNan(float a, float b) {
    return isnan(a) || isnan(b) ? asfloat(0x7fffffffu) : min(a, b);
}

//! An FP32 value to a 10-bit unorm channel: clamp, truncate to 14 fraction bits, then scale with ties down.
public uint unorm10(float value) {
    uint fixed = uint(floor(saturate(value) * 16384.0f));
    return (fixed * 1023u + 8191u) >> 14;
}

//! FP32 to FP16 with round toward zero (finite values saturate to 65504), returned as FP32. Integer arithmetic, so that no compiler removes the rounding.
public float roundTowardZeroHalf(float value) {
    uint bits = asuint(value);
    uint sign = bits & 0x80000000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u)
        return value;
    if (magnitude >= 0x477ff000u)
        return asfloat(sign | 0x477fe000u);
    if (magnitude < 0x33800000u)
        return asfloat(sign);
    if (magnitude < 0x38800000u) {
        uint mantissa = (magnitude & 0x007fffffu) | 0x00800000u;
        float truncated = float(mantissa >> (126u - (magnitude >> 23))) * 5.9604644775390625e-08f;
        return sign != 0u ? -truncated : truncated;
    }
    return asfloat(sign | (magnitude & 0xffffe000u));
}

//! Round to nearest with ties to even. Values of 2^23 and more are integers already.
public float roundNearestEven(float value) {
    float magnitude = abs(value);
    float rounded = magnitude < 8388608.0f ? (magnitude + 8388608.0f) - 8388608.0f : magnitude;
    return value < 0.0f ? -rounded : rounded;
}
