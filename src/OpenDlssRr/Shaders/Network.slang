//! The functions that the network passes share. docs/network.md describes the network.
//! Activations are FP16 in the layout [y][x][channel]; the weights are one FP32 array, each tensor row major in its source shape.
module Network;

//! The side of an attention window: 8x8 = 64 tokens.
public static const int window = 8;
//! The channels of one attention head (queries, keys and values).
public static const int headSize = 32;

public float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

//! The GELU of the network: a quadratic blend between -2 and 2, 0 below, x above.
public half gelu(half x) {
    half t = clamp(x, half(-2.0f), half(2.0f));
    return x * (half(0.5f) + t * (half(0.412162f) - half(0.081081f) * abs(t)));
}

//! The offset of the 16x16 tile at (row, column) of a matrix with `columns` columns in the FP16 weights, which hold each matrix in tiles
//! of 256 contiguous halves (row major in the tile, tiles row major); a load of the tile uses the stride 16.
public uint tileAt(uint matrix, int row, int column, int columns) {
    return matrix + uint((row / 16 * (columns / 16) + column / 16) * 256);
}

//! FP32 to FP16 bits with the hardware conversion (its rounding mode depends on the GPU).
public uint halfBits(float value) {
    return uint(asuint16(half(value)));
}

//! Mirror padding without a repeat of the edge pixel: -1 -> 1, size -> size - 2.
public int reflect(int index, int size) {
    index = abs(index);
    return min(index, 2 * size - 2 - index);
}

//! The 2x up-sampling of a colour along one axis (sample centres at 2i + 0.5): pixel X reads (X - 1) / 2 and the next pixel, with the
//! weight 0.25 (X odd) or 0.75 (X even) on the second one; pixel 0 copies pixel 0, and the last pixel clamps.
public struct UpAxis {
    public int first;
    public int second;
    public float fraction;
    public __init(int index, int coarseSize) {
        first = index >= 1 ? (index - 1) / 2 : 0;
        second = min(first + 1, coarseSize - 1);
        fraction = index == 0 ? 0.0f : (index % 2 == 1 ? 0.25f : 0.75f);
    }
}

//! Bilinear 2x up-sampling of a colour image of the coarser level (coarseSize pixels, FP16 RGBA per pixel) at pixel p of this level.
public float4 upColour(StructuredBuffer<half4> coarse, int2 coarseSize, int2 p) {
    UpAxis x = UpAxis(p.x, coarseSize.x);
    UpAxis y = UpAxis(p.y, coarseSize.y);
    float4 a = float4(coarse[y.first * coarseSize.x + x.first]);
    float4 b = float4(coarse[y.first * coarseSize.x + x.second]);
    float4 c = float4(coarse[y.second * coarseSize.x + x.first]);
    float4 d = float4(coarse[y.second * coarseSize.x + x.second]);
    return (1.0f - x.fraction) * (1.0f - y.fraction) * a + x.fraction * (1.0f - y.fraction) * b + (1.0f - x.fraction) * y.fraction * c
           + x.fraction * y.fraction * d;
}
