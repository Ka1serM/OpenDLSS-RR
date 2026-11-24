//! The second half of a transformer block: the projection of the attention output with a scaled residual, then the MLP C -> 4C -> C with
//! a scaled residual. These are per pixel, so a workgroup computes one tile of 16 consecutive pixels, with 8 subgroups that split the
//! output columns of the projection and then the hidden layer. The levels are small, so more workgroups (each streams all the weights
//! from the cache) are faster than fewer that share them. The activation buffers have room for a multiple of 64 pixels, so the tiles load
//! and store directly. With Block.slang at stage 2 (the attention, one workgroup per window and head), this splits a block over more
//! workgroups than one per window, for the levels with few windows.
import Network;

typealias MatrixA = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias MatrixB = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias Sum = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;
typealias Product = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;
static const linalg.CoopMatMatrixLayout rowMajor = linalg.CoopMatMatrixLayout.RowMajor;
static const linalg.CoopMatMatrixLayout columnMajor = linalg.CoopMatMatrixLayout.ColumnMajor;

//! The block input, the weights (FP16 matrices, FP32 biases, at the same offsets), the block output and the attention output, each
//! pixel after the other with all its channels.
[[vk::binding(0)]] StructuredBuffer<half> input;
[[vk::binding(1)]] StructuredBuffer<half> matrices;
[[vk::binding(2)]] StructuredBuffer<float> weights;
[[vk::binding(3)]] RWStructuredBuffer<half> output;
[[vk::binding(4)]] StructuredBuffer<half> attended;
//! The colour head (as in Block.slang): the gated colour of the finer level (gate) or of this level (blend), the colour of the coarser
//! level, and the colour output.
[[vk::binding(5)]] StructuredBuffer<half4> colour;
[[vk::binding(6)]] StructuredBuffer<half4> coarse;
[[vk::binding(7)]] RWStructuredBuffer<half4> colourOutput;

//! The push constants of Block.slang; this kernel reads the projection, the MLP and the colour head.
struct Constants {
    int2 size;
    int2 shift;
    uint qkv;
    uint pairs;
    uint projection;
    uint projectionBias;
    uint hidden;
    uint hiddenBias;
    uint mlp;
    uint mlpBias;
    float attentionScale;
    float mlpScale;
    uint prologueWeight;
    uint prologueBias;
    uint stemOutput;
    uint stemOutputBias;
    float skipScale;
    //! The colour head: W (3 or 4 x channels), b and the gain.
    uint headWeight;
    uint headBias;
    float gain;
};
[[vk::push_constant]] ConstantBuffer<Constants> constants;

[SpecializationConstant] const int channels = 160;
[SpecializationConstant] const int heads = 8;
//! The colour head after the block (as in Block.slang): none, the gate of the 2x2 mean of the finer gated colour, or the blend.
[SpecializationConstant] const int colourHead = 0;
static const int headGatePooled = 2;
static const int headBlend = 3;

static const int lanes = 32;
static const int subgroups = 8;
static const int threads = subgroups * lanes;
static const int tilesC = channels / 16;
static const int attentionChannels = heads * headSize;
static const int hiddenChannels = 4 * channels;
//! The column tiles of the projection per subgroup.
static const int projectionTiles = (tilesC + subgroups - 1) / subgroups;

//! The hidden channels of one step of the MLP.
static const int step = 16;

//! The block input of the tile, then per subgroup two buffers of one step of the hidden layer (16 x step each): a step computes the next
//! hidden channels into one buffer while it reads the other. After the hidden layer, the same memory holds the partial sums of the second
//! layer (16 x C each) of half the subgroups. Row strides in halves, padded by 8 against bank conflicts.
static const int xStride = channels + 8;
static const int stepStride = step + 8;
static const int scratchAt = 16 * xStride;
static const int bufferSize = 16 * stepStride;
static const int partialSize = 16 * xStride;
static const int scratchSize = subgroups * 2 * bufferSize > subgroups / 2 * partialSize ? subgroups * 2 * bufferSize : subgroups / 2 * partialSize;
groupshared half memory[scratchAt + scratchSize];

//! A broadcast row of 16 FP32 values (a bias) as a 16x16 FP16 sum.
Product loadBias(uint at) {
    return Product(Sum.Load<rowMajor>(weights, at, 0));
}

void subgroupSync() {
    GroupMemoryBarrierWithWaveSync();
}

//! GELU(x W1 + b1) for the hidden channels h ... h + step - 1, to shared memory at `at`.
void hiddenStep(MatrixA x[tilesC], int h, int at) {
    Product hidden[step / 16];
    [unroll]
    for (int t = 0; t < step / 16; ++t)
        hidden[t] = loadBias(constants.hiddenBias + h + t * 16);
    [unroll]
    for (int k = 0; k < tilesC; ++k)
        [unroll]
        for (int t = 0; t < step / 16; ++t)
            hidden[t] = linalg.coopMatMulAdd<half, false>(x[k], MatrixB.Load<columnMajor>(matrices, tileAt(constants.hidden, h + t * 16, k * 16, channels), 16), hidden[t]);
    [unroll]
    for (int t = 0; t < step / 16; ++t) {
        for (int e = 0; e < hidden[t].getCount(); ++e)
            hidden[t][e] = gelu(hidden[t][e]);
        hidden[t].Store<rowMajor>(memory, at + t * 16, stepStride);
    }
}

[numthreads(threads, 1, 1)]
[shader("compute")]
void mlp(uint3 group : SV_GroupID, uint thread : SV_GroupIndex)
{
    int subgroup = int(thread / lanes);
    uint first = group.x * 16;
    int buffers = scratchAt + subgroup * 2 * bufferSize;

    //! x + attentionScale · (projection of the attention output + bias), for the column tiles subgroup, subgroup + 8, ...
    {
        Product sums[projectionTiles];
        [unroll]
        for (int j = 0; j < projectionTiles; ++j)
            sums[j] = loadBias(constants.projectionBias + min(j * subgroups + subgroup, tilesC - 1) * 16);
        [unroll]
        for (int k = 0; k < attentionChannels; k += 16) {
            MatrixA a = MatrixA.Load<rowMajor>(attended, first * attentionChannels + k, attentionChannels);
            [unroll]
            for (int j = 0; j < projectionTiles; ++j)
                if (j * subgroups + subgroup < tilesC)
                    sums[j] = linalg.coopMatMulAdd<half, false>(a, MatrixB.Load<columnMajor>(matrices, tileAt(constants.projection, (j * subgroups + subgroup) * 16, k, attentionChannels), 16), sums[j]);
        }
        [unroll]
        for (int j = 0; j < projectionTiles; ++j) {
            int column = j * subgroups + subgroup;
            if (column >= tilesC)
                continue;
            Product x = Product.Load<rowMajor>(input, first * channels + column * 16, channels);
            for (int e = 0; e < x.getCount(); ++e)
                x[e] = x[e] + half(constants.attentionScale) * sums[j][e];
            x.Store<rowMajor>(memory, column * 16, xStride);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    //! GELU(x W1 + b1) for one eighth of the hidden layer, step channels at a time, and its part of the second layer: the next step of
    //! the hidden layer runs before the second layer of the current one, so neither waits for the other.
    MatrixA x[tilesC];
    [unroll]
    for (int k = 0; k < tilesC; ++k)
        x[k] = MatrixA.Load<rowMajor>(memory, k * 16, xStride);
    Product sums[tilesC];
    [unroll]
    for (int j = 0; j < tilesC; ++j)
        sums[j] = subgroup == 0 ? loadBias(constants.mlpBias + j * 16) : Product(half(0.0f));
    int begin = subgroup * hiddenChannels / subgroups;
    int end = begin + hiddenChannels / subgroups;
    hiddenStep(x, begin, buffers);
    subgroupSync();
    for (int h = begin; h < end; h += step) {
        int current = buffers + (h - begin) / step % 2 * bufferSize;
        MatrixA a[step / 16];
        [unroll]
        for (int t = 0; t < step / 16; ++t)
            a[t] = MatrixA.Load<rowMajor>(memory, current + t * 16, stepStride);
        if (h + step < end)
            hiddenStep(x, h + step, buffers + ((h - begin) / step + 1) % 2 * bufferSize);
        [unroll]
        for (int t = 0; t < step / 16; ++t)
            [unroll]
            for (int j = 0; j < tilesC; ++j)
                sums[j] = linalg.coopMatMulAdd<half, false>(a[t], MatrixB.Load<columnMajor>(matrices, tileAt(constants.mlp, j * 16, h + t * 16, hiddenChannels), 16), sums[j]);
        subgroupSync();
    }

    //! The sum of the parts, in halving rounds: the upper half of the subgroups stores, the lower half adds.
    for (int count = subgroups / 2; count > 0; count /= 2) {
        GroupMemoryBarrierWithGroupSync();
        if (subgroup >= count && subgroup < 2 * count)
            [unroll]
            for (int j = 0; j < tilesC; ++j)
                sums[j].Store<rowMajor>(memory, scratchAt + (subgroup - count) * partialSize + j * 16, xStride);
        GroupMemoryBarrierWithGroupSync();
        if (subgroup < count)
            [unroll]
            for (int j = 0; j < tilesC; ++j) {
                Product part = Product.Load<rowMajor>(memory, scratchAt + subgroup * partialSize + j * 16, xStride);
                for (int e = 0; e < part.getCount(); ++e)
                    sums[j][e] = sums[j][e] + part[e];
            }
    }
    if (subgroup != 0)
        return;

    //! x + mlpScale · (MLP + bias).
    [unroll]
    for (int j = 0; j < tilesC; ++j) {
        Product x = Product.Load<rowMajor>(memory, j * 16, xStride);
        for (int e = 0; e < x.getCount(); ++e)
            x[e] = x[e] + half(constants.mlpScale) * sums[j][e];
        x.Store<rowMajor>(memory, j * 16, xStride);
        x.Store<rowMajor>(output, first * channels + j * 16, channels);
    }
    subgroupSync();
    if (colourHead == 0 || thread >= 16)
        return;

    //! The colour head of a pixel: logits = W · x + b, then the gate or the blend.
    uint pixel = first + thread;
    if (pixel >= uint(constants.size.x * constants.size.y))
        return;
    int2 p = int2(int(pixel) % constants.size.x, int(pixel) / constants.size.x);
    int logitCount = colourHead == headBlend ? 4 : 3;
    float logits[4];
    for (int i = 0; i < logitCount; ++i)
        logits[i] = weights[constants.headBias + i];
    for (int c = 0; c < channels; ++c) {
        float value = float(memory[thread * xStride + c]);
        [unroll]
        for (int i = 0; i < logitCount; ++i)
            logits[i] += value * weights[constants.headWeight + i * channels + c];
    }
    if (colourHead == headBlend) {
        float alpha = sigmoid(logits[3]);
        float3 up = upColour(coarse, constants.size / 2, p).rgb;
        float3 scaled = up * constants.gain * float3(sigmoid(logits[0]), sigmoid(logits[1]), sigmoid(logits[2]));
        float4 blended = float4(colour[pixel]) * (1.0f - alpha) + float4(scaled, 0.0f) * alpha;
        colourOutput[pixel] = half4(min(blended, 65504.0f));
        return;
    }
    int width = 2 * constants.size.x;
    int at = 2 * p.y * width + 2 * p.x;
    float4 pooled = (float4(colour[at]) + float4(colour[at + 1]) + float4(colour[at + width]) + float4(colour[at + width + 1])) * 0.25f;
    pooled = float4(half4(pooled));
    float3 gated;
    [unroll]
    for (int i = 0; i < 3; ++i)
        gated[i] = min(pooled[i] * constants.gain * sigmoid(logits[i]), 65504.0f);
    colourOutput[pixel] = half4(half3(gated), 0.0h);
}
