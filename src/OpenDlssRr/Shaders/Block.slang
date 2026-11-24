//! One transformer block for one 8x8 window of 64 tokens, in one workgroup of 4 subgroups of 32 lanes: windowed attention with a scaled
//! residual, then the MLP C -> 4C -> C with a scaled residual. The matrix products use cooperative matrices (16x16x16, FP16 inputs and
//! sums). In each product, a subgroup owns whole column tiles across rows of 16 tokens, so the workgroup loads each weight tile only once.
//! The first block of a level can also compute its input (the prologue): enc0 from the 48 features of each token (the stem 48 -> 64 -> 32
//! with a GELU), enc1 ... enc5 with the down convolution of the finer level (2x2, stride 2), dec0 ... dec4 with the up convolution of the
//! coarser level (2x2 transposed, stride 2) plus the scaled skip (the encoder output of the level, read as the block input).
//! The tokens of a window are in the order of their 2x2 parity (2 (y % 2) + x % 2), then of their 4x4 position: so the 16 tokens of a row
//! tile use one weight of the up convolution. The attention bias tables are in this order too.
import Network;
import Features;

typealias MatrixA = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias MatrixB = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias Sum = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;
typealias Product = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;
static const linalg.CoopMatMatrixLayout rowMajor = linalg.CoopMatMatrixLayout.RowMajor;
static const linalg.CoopMatMatrixLayout columnMajor = linalg.CoopMatMatrixLayout.ColumnMajor;

//! The block input, the features of the stem or the skip of the up convolution.
[[vk::binding(0)]] StructuredBuffer<uint4> input;
//! The weights twice, at the same offsets: FP16 for the matrix products and the attention bias tables, FP32 for the biases.
[[vk::binding(1)]] StructuredBuffer<half> matrices;
[[vk::binding(2)]] StructuredBuffer<float> weights;
[[vk::binding(3)]] RWStructuredBuffer<uint4> output;
//! The input of the down or the up convolution: the finer or the coarser level, sourceChannels per pixel.
[[vk::binding(4)]] StructuredBuffer<uint4> source;
//! The colour head: the colour to gate (the input colour or the finer gated colour) or the gated colour to blend, the coarser colour of
//! the blend, and the result.
[[vk::binding(5)]] StructuredBuffer<half4> colour;
[[vk::binding(6)]] StructuredBuffer<half4> coarse;
[[vk::binding(7)]] RWStructuredBuffer<half4> colourOutput;
//! The images of the output heads (Output.slang), of the render size.
[[vk::binding(8)]] [[vk::image_format("rgba8_snorm")]] RWTexture2D<float4> feedback;
[[vk::binding(9)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> guide;
[[vk::binding(10)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> candidateWeights;
[[vk::binding(11)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> shape;
//! The input of the next encoder: the down convolution of the block output, nextChannels per pixel of half this size.
[[vk::binding(12)]] RWStructuredBuffer<uint4> next;

struct Constants {
    int2 size;
    int2 shift;
    //! The tensors: QKV (2 · 32 heads x C, no bias), the bias tables, the projection (C x 32 heads) and its bias, the MLP layers and their biases.
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
    //! The prologue: the first stem layer or the convolution, and its bias; the second stem layer and its bias; the scale of the skip.
    uint prologueWeight;
    uint prologueBias;
    uint stemOutput;
    uint stemOutputBias;
    float skipScale;
    //! The colour head: W (3 or 4 x C), b and the gain.
    uint headWeight;
    uint headBias;
    float gain;
    //! The output heads: the render size and exp(dec0_1.logCorrelation).
    int2 renderSize;
    float correlationScale;
    //! The down convolution of the next encoder: W (nextChannels x 4 x C) and its bias.
    uint nextWeight;
    uint nextBias;
};
[[vk::push_constant]] ConstantBuffer<Constants> constants;

//! The channels C and the attention heads of the level.
[SpecializationConstant] const int channels = 32;
[SpecializationConstant] const int heads = 2;
//! The split of the products with C outputs: rowGroups groups of 4 / rowGroups subgroups, colsPerSubgroup column tiles per subgroup.
[SpecializationConstant] const int rowGroups = 1;
[SpecializationConstant] const int colsPerSubgroup = 1;
//! The prologue, and the channels per pixel of the convolution input.
[SpecializationConstant] const int prologue = 0;
[SpecializationConstant] const int sourceChannels = 32;
static const int prologueStem = 1;
static const int prologueDown = 2;
static const int prologueUp = 3;
//! Another prologue reads the input like none: it is the down convolution that the finer encoder block computed (nextChannels).
//! What the dispatch computes: the whole block; only the block input (the prologue) to output, the column tiles of part group.z; or
//! only the attention of head group.z, its 32 channels to output with heads x 32 channels per pixel (Mlp.slang then does the rest of the
//! block).
[SpecializationConstant] const int stage = 0;
static const int stageBlock = 0;
static const int stageInput = 1;
static const int stageAttention = 2;
//! The colour head after the block, with logits W · x + b: none; the gate (colour · gain · sigmoid(logits 0 ... 2), alpha 0) of the colour
//! of this size or of the 2x2 mean of the finer gated colour; the blend (alpha = sigmoid(logit 3) mixes the gated colour of this level and
//! the up-sampled colour of the coarser level times gain · sigmoid(logits 0 ... 2)); or the output heads of dec0_1, which replace the block
//! output.
[SpecializationConstant] const int colourHead = 0;
static const int headGate = 1;
static const int headGatePooled = 2;
static const int headBlend = 3;
static const int headOutput = 4;
//! The channels of the next encoder, whose input this block computes, or 0.
[SpecializationConstant] const int nextChannels = 0;
//! The standard deviations of the post filter are at most 8 pixels.
static const float sigmaScale = 8.0f;

static const int lanes = 32;
static const int threads = 4 * lanes;
static const int tokens = window * window;
static const int tilesC = channels / 16;
static const int colGroups = 4 / rowGroups;
static const int rowsPerSubgroup = 4 / rowGroups;
static const int attentionChannels = heads * headSize;
static const int words = channels / 8;
static const int featureWords = 6;
static const int sourceWords = sourceChannels / 8;
static const int stemWidth = 64;

//! Row strides in halves, padded against bank conflicts: the block input and then its results (tokens x C); the queries and values of
//! one head (tokens x 32 each). After the scores, the queries of a row tile hold its 16x16 tiles of attention weights, then its attention
//! output. The queries and values also hold 64 channels of the MLP hidden layer or of the convolution input (tokens x 64), or the
//! features of the stem; after them, the biases of the up convolution (4 x C). Less shared memory fits more workgroups on a multiprocessor.
static const int xStride = channels + 8;
static const int headStride = headSize + 8;
static const int wideStride = 64 + 8;
static const int queryAt = tokens * xStride;
static const int valueAt = queryAt + tokens * headStride;
//! The bias of the up convolution, after the source staging of the prologue.
static const int biasAt = queryAt + tokens * wideStride;
static const int memorySize = valueAt + tokens * headStride > biasAt + 4 * channels ? valueAt + tokens * headStride : biasAt + 4 * channels;
static const int featureStride = 8 * featureWords + 8;
groupshared half memory[memorySize];

//! 2^e with e = 32 (x (0.927 - x^2) + 0.375), x the clamped scaled score, times 2^-15: y = 1 + e / 32 is in [1, 2), so the 5 high bits
//! of its FP16 mantissa are the integer part of e and the 5 low bits its fraction, which become the exponent and the mantissa.
half attentionWeight(half score) {
    half x = clamp(score * half(0.09722317f / sqrt(float(headSize))), half(-0.55615234f), half(0.55615234f));
    half y = x * (half(0.92724609f) - x * x) + half(1.375f);
    return asfloat16(uint16_t((uint(asuint16(y)) << 5) & 0x7fe0));
}

//! A window reaches at most 7 tokens past the image, which is larger than that.
int mirror(int index, int size) {
    index = abs(index);
    return min(index, 2 * size - 2 - index);
}

int2 tokenPosition(int2 origin, int token) {
    int parity = token / 16;
    int index = token % 16;
    return origin + int2(2 * (index % 4) + parity % 2, 2 * (index / 4) + parity / 2);
}

//! 8 halves from a 16-byte word.
void unpackTo(uint4 word, int at) {
    [unroll]
    for (int i = 0; i < 4; ++i) {
        memory[at + 2 * i] = half(f16tof32(word[i]));
        memory[at + 2 * i + 1] = half(f16tof32(word[i] >> 16));
    }
}

void subgroupSync() {
    GroupMemoryBarrierWithWaveSync();
}

//! A broadcast row of 16 FP32 values (a bias) as a 16x16 FP16 sum.
Product loadBias(uint at) {
    return Product(Sum.Load<rowMajor>(weights, at, 0));
}

//! The tokens that the window owns to output: `count` channels per token at `at` with a row stride, and `pixelWords` words per pixel
//! from word `first`.
void store(int2 origin, uint thread, int at, int stride, int count, int pixelWords, int first) {
    for (int index = int(thread); index < tokens * count / 8; index += threads) {
        int token = index / (count / 8);
        int word = index % (count / 8);
        int2 p = tokenPosition(origin, token);
        if (any(p < 0) || any(p >= constants.size))
            continue;
        int from = at + token * stride + word * 8;
        uint4 bits;
        [unroll]
        for (int i = 0; i < 4; ++i)
            bits[i] = halfBits(float(memory[from + 2 * i])) | halfBits(float(memory[from + 2 * i + 1])) << 16;
        output[(p.y * constants.size.x + p.x) * pixelWords + first + word] = bits;
    }
}

//! The row tiles and the column tiles of C that a subgroup owns; the column tiles from columnBase, the first of the part of stage Input.
static int columnBase = 0;
static const int partTiles = colGroups * colsPerSubgroup;

int ownedRow(uint subgroup, int i) {
    return int(subgroup) / colGroups * rowsPerSubgroup + i;
}

int ownedColumn(uint subgroup, int j) {
    return columnBase + int(subgroup) % colGroups + j * colGroups;
}

//! The block; with `fused` (the stem), the features come from Features.slang, 2 lanes per token, and the lane of the enc0 colour head
//! keeps the exposed colour of its token.
void run<let fused : bool>(uint3 group, uint thread)
{
    uint subgroup = thread / lanes;
    int2 origin = int2(group.xy) * window - constants.shift;
    if (stage == stageInput)
        columnBase = int(group.z) * partTiles;

    float3 exposed = float3(0.0f);
    if (fused) {
        int token = int(subgroup) * 16 + int(thread % 16);
        int part = int(thread % lanes) / 16;
        int2 p = tokenPosition(origin, token);
        int2 grid = int2(mirror(p.x, constants.size.x), mirror(p.y, constants.size.y));
        float values[24];
        features(grid, part, uint(weights[noiseAt(grid)]), all(p >= 0) && all(p < constants.size), values, exposed);
        [unroll]
        for (int c = 0; c < 24; ++c)
            memory[queryAt + token * featureStride + part * 24 + c] = half(values[c]);
    }

    //! The block input (the features, the skip), 8 channels per 16-byte word, mirrored at the image edges.
    int inputWords = prologue == prologueStem ? featureWords : words;
    int inputAt = prologue == prologueStem ? queryAt : 0;
    int inputStride = prologue == prologueStem ? featureStride : xStride;
    if (prologue != prologueDown && !fused)
        for (int index = int(thread); index < tokens * inputWords; index += threads) {
            int token = index / inputWords;
            int2 p = tokenPosition(origin, token);
            int at = mirror(p.y, constants.size.y) * constants.size.x + mirror(p.x, constants.size.x);
            unpackTo(input[at * inputWords + index % inputWords], inputAt + token * inputStride + (index % inputWords) * 8);
        }
    //! For up with no mirrored token, the 16 tokens of each row tile are the 16 coarser pixels under the window: they go to shared memory
    //! now, with all their channels, and every row tile reads them.
    bool inside = prologue == prologueUp && all(origin >= 0) && all(origin + window <= constants.size);
    if (inside)
        for (int index = int(thread); index < 16 * sourceWords; index += threads) {
            int token = index / sourceWords;
            int2 s = tokenPosition(origin, token) / 2;
            unpackTo(source[(s.y * (constants.size.x / 2) + s.x) * sourceWords + index % sourceWords],
                     queryAt + token * (sourceChannels + 8) + (index % sourceWords) * 8);
        }
    //! The bias of the up convolution is (output channel, parity); its rows go to shared memory by parity.
    if (prologue == prologueUp)
        for (int index = int(thread); index < 4 * channels; index += threads)
            memory[biasAt + index] = half(weights[constants.prologueBias + (index % channels) * 4 + index / channels]);
    GroupMemoryBarrierWithGroupSync();

    if (prologue == prologueStem) {
        //! GELU(features W1 + b1): the subgroup computes one of the 4 column tiles for all tokens.
        Product hidden[4];
        [unroll]
        for (int r = 0; r < 4; ++r)
            hidden[r] = loadBias(constants.prologueBias + subgroup * 16);
        [unroll]
        for (int k = 0; k < 8 * featureWords; k += 16) {
            MatrixB b = MatrixB.Load<columnMajor>(matrices, tileAt(constants.prologueWeight, int(subgroup) * 16, k, stemWidth), 16);
            [unroll]
            for (int r = 0; r < 4; ++r)
                hidden[r] = linalg.coopMatMulAdd<half, false>(MatrixA.Load<rowMajor>(memory, queryAt + r * 16 * featureStride + k, featureStride), b, hidden[r]);
        }
        [unroll]
        for (int r = 0; r < 4; ++r)
            for (int e = 0; e < hidden[r].getCount(); ++e)
                hidden[r][e] = gelu(hidden[r][e]);
        //! The block input: hidden W2 + b2, with the hidden layer in shared memory in place of the block input, 32 channels at a time.
        Product x[rowsPerSubgroup][colsPerSubgroup];
        [unroll]
        for (int j = 0; j < colsPerSubgroup; ++j)
            [unroll]
            for (int i = 0; i < rowsPerSubgroup; ++i)
                x[i][j] = loadBias(constants.stemOutputBias + min(ownedColumn(subgroup, j), tilesC - 1) * 16);
        [unroll]
        for (int part = 0; part < 2; ++part) {
            if (int(subgroup) / 2 == part)
                [unroll]
                for (int r = 0; r < 4; ++r)
                    hidden[r].Store<rowMajor>(memory, r * 16 * xStride + int(subgroup) % 2 * 16, xStride);
            GroupMemoryBarrierWithGroupSync();
            [unroll]
            for (int k = 0; k < 32; k += 16) {
                MatrixA a[rowsPerSubgroup];
                [unroll]
                for (int i = 0; i < rowsPerSubgroup; ++i)
                    a[i] = MatrixA.Load<rowMajor>(memory, ownedRow(subgroup, i) * 16 * xStride + k, xStride);
                [unroll]
                for (int j = 0; j < colsPerSubgroup; ++j) {
                    int column = ownedColumn(subgroup, j);
                    if (column >= tilesC)
                        continue;
                    MatrixB b = MatrixB.Load<columnMajor>(matrices, tileAt(constants.stemOutput, column * 16, part * 32 + k, stemWidth), 16);
                    [unroll]
                    for (int i = 0; i < rowsPerSubgroup; ++i)
                        x[i][j] = linalg.coopMatMulAdd<half, false>(a[i], b, x[i][j]);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        [unroll]
        for (int j = 0; j < colsPerSubgroup; ++j) {
            int column = ownedColumn(subgroup, j);
            if (column >= tilesC)
                continue;
            [unroll]
            for (int i = 0; i < rowsPerSubgroup; ++i)
                x[i][j].Store<rowMajor>(memory, ownedRow(subgroup, i) * 16 * xStride + column * 16, xStride);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (prologue == prologueDown || prologue == prologueUp) {
        //! The convolution: W is outputChannels x 4 x sourceChannels, and the bias is per output channel (down) or per output channel and
        //! parity (up). Down: a token reads 4 finer pixels, one after the other. Up: it reads the coarser pixel under it, with the weight
        //! of its parity, which is the row tile. The source goes through shared memory in parts of 64 channels, unless it is already there.
        Product x[rowsPerSubgroup][colsPerSubgroup];
        [unroll]
        for (int j = 0; j < colsPerSubgroup; ++j)
            [unroll]
            for (int i = 0; i < rowsPerSubgroup; ++i) {
                int column = min(ownedColumn(subgroup, j), tilesC - 1);
                x[i][j] = prologue == prologueDown ? loadBias(constants.prologueBias + column * 16)
                                                   : Product.Load<rowMajor>(memory, biasAt + ownedRow(subgroup, i) * channels + column * 16, 0);
            }
        int positions = prologue == prologueDown ? 4 : 1;
        int sourceWidth = prologue == prologueDown ? 2 * constants.size.x : constants.size.x / 2;
        int partChannels = inside ? sourceChannels : 64;
        int stride = partChannels + 8;
        for (int position = 0; position < positions; ++position)
            for (int part = 0; part < sourceChannels; part += partChannels) {
                int partWords = min(partChannels / 8, sourceWords - part / 8);
                if (!inside) {
                    for (int index = int(thread); index < tokens * partWords; index += threads) {
                        int token = index / partWords;
                        int2 p = tokenPosition(origin, token);
                        int2 q = int2(mirror(p.x, constants.size.x), mirror(p.y, constants.size.y));
                        int2 s = prologue == prologueDown ? 2 * q + int2(position % 2, position / 2) : q / 2;
                        unpackTo(source[(s.y * sourceWidth + s.x) * sourceWords + part / 8 + index % partWords],
                                 queryAt + token * stride + (index % partWords) * 8);
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
                for (int k = 0; k < 8 * partWords; k += 16) {
                    MatrixA first = MatrixA.Load<rowMajor>(memory, queryAt + (inside ? 0 : ownedRow(subgroup, 0)) * 16 * stride + k, stride);
                    MatrixA a[rowsPerSubgroup];
                    [unroll]
                    for (int i = 0; i < rowsPerSubgroup; ++i)
                        a[i] = i == 0 || inside ? first : MatrixA.Load<rowMajor>(memory, queryAt + ownedRow(subgroup, i) * 16 * stride + k, stride);
                    [unroll]
                    for (int j = 0; j < colsPerSubgroup; ++j) {
                        int column = ownedColumn(subgroup, j);
                        if (column >= tilesC)
                            continue;
                        [unroll]
                        for (int i = 0; i < rowsPerSubgroup; ++i) {
                            int weightPosition = prologue == prologueDown ? position : ownedRow(subgroup, i);
                            uint at = tileAt(constants.prologueWeight, column * 16, weightPosition * sourceChannels + part + k, 4 * sourceChannels);
                            x[i][j] = linalg.coopMatMulAdd<half, false>(a[i], MatrixB.Load<columnMajor>(matrices, at, 16), x[i][j]);
                        }
                    }
                }
                if (!inside)
                    GroupMemoryBarrierWithGroupSync();
            }
        //! The block input: the convolution, plus the scaled skip for up.
        [unroll]
        for (int j = 0; j < colsPerSubgroup; ++j) {
            int column = ownedColumn(subgroup, j);
            if (column >= tilesC)
                continue;
            [unroll]
            for (int i = 0; i < rowsPerSubgroup; ++i) {
                int at = ownedRow(subgroup, i) * 16 * xStride + column * 16;
                if (prologue == prologueUp) {
                    Product skip = Product.Load<rowMajor>(memory, at, xStride);
                    for (int e = 0; e < skip.getCount(); ++e)
                        x[i][j][e] = x[i][j][e] + half(constants.skipScale) * skip[e];
                }
                x[i][j].Store<rowMajor>(memory, at, xStride);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (stage == stageInput) {
        store(origin, thread, columnBase * 16, xStride, partTiles * 16, words, columnBase * 2);
        return;
    }

    Product sums[rowsPerSubgroup][colsPerSubgroup];
    [unroll]
    for (int j = 0; j < colsPerSubgroup; ++j)
        [unroll]
        for (int i = 0; i < rowsPerSubgroup; ++i)
            sums[i][j] = loadBias(constants.projectionBias + min(ownedColumn(subgroup, j), tilesC - 1) * 16);

    int firstHead = stage == stageAttention ? int(group.z) : 0;
    Product mean[2];
    for (int head = firstHead; head < (stage == stageAttention ? firstHead + 1 : heads); ++head) {
        //! The queries and values of the head: the subgroup computes one of the 4 column tiles (2 of queries, 2 of values) for all tokens.
        {
            uint part = subgroup / 2;
            uint column = subgroup % 2;
            uint weightRow = (part * heads + head) * headSize + column * 16;
            Product values[4];
            [unroll]
            for (int r = 0; r < 4; ++r)
                values[r] = Product(half(0.0f));
            [unroll]
            for (int k = 0; k < channels; k += 16) {
                MatrixB b = MatrixB.Load<columnMajor>(matrices, tileAt(constants.qkv, int(weightRow), k, channels), 16);
                [unroll]
                for (int r = 0; r < 4; ++r)
                    values[r] = linalg.coopMatMulAdd<half, false>(MatrixA.Load<rowMajor>(memory, r * 16 * xStride + k, xStride), b, values[r]);
            }
            uint target = (part != 0 ? valueAt : queryAt) + column * 16;
            //! The projection of the head before reads its attention output in place of the queries.
            if (head != firstHead)
                GroupMemoryBarrierWithGroupSync();
            [unroll]
            for (int r = 0; r < 4; ++r)
                values[r].Store<rowMajor>(memory, target + r * 16 * headStride, headStride);
        }
        GroupMemoryBarrierWithGroupSync();

        //! The attention of the 16 tokens of row tile `subgroup`: the weights P of the token pairs (score = q_i · q_j + bias[i][j]), then
        //! the weighted mean of the values: P divided by its row sums P 1, then P V, one column tile of P at a time through shared memory.
        //! The row sums have the layout of P: each element is the sum of its row.
        {
            int row = int(subgroup);
            int at = queryAt + row * 16 * headStride;
            MatrixA query[2] = { MatrixA.Load<rowMajor>(memory, queryAt + row * 16 * headStride, headStride),
                                 MatrixA.Load<rowMajor>(memory, queryAt + row * 16 * headStride + 16, headStride) };
            Product scores[4];
            [unroll]
            for (int j = 0; j < 4; ++j) {
                scores[j] = Product.Load<rowMajor>(matrices, tileAt(constants.pairs, head * tokens + row * 16, j * 16, tokens), 16);
                [unroll]
                for (int k = 0; k < 2; ++k)
                    scores[j] = linalg.coopMatMulAdd<half, false>(query[k], MatrixB.Load<columnMajor>(memory, queryAt + j * 16 * headStride + k * 16, headStride), scores[j]);
            }
            //! P goes through shared memory in place of the queries of the row tile, when no subgroup reads them any more; P 1 and P V
            //! accumulate in one pass, and P V is divided by P 1 at the end. P is scaled by 2^-8 so that P V stays in the FP16 range.
            GroupMemoryBarrierWithGroupSync();
            Product total = Product(half(0.0f));
            MatrixB ones = MatrixB(half(1.0f));
            mean[0] = Product(half(0.0f));
            mean[1] = Product(half(0.0f));
            [unroll]
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < scores[j].getCount(); ++i)
                    scores[j][i] = attentionWeight(scores[j][i]) * half(1.0f / 256.0f);
                scores[j].Store<rowMajor>(memory, at, headStride);
                subgroupSync();
                MatrixA a = MatrixA.Load<rowMajor>(memory, at, headStride);
                total = linalg.coopMatMulAdd<half, false>(a, ones, total);
                [unroll]
                for (int column = 0; column < 2; ++column)
                    mean[column] = linalg.coopMatMulAdd<half, false>(a, MatrixB.Load<rowMajor>(memory, valueAt + j * 16 * headStride + column * 16, headStride), mean[column]);
                subgroupSync();
            }
            for (int i = 0; i < total.getCount(); ++i) {
                half scale = half(1.0f) / total[i];
                mean[0][i] *= scale;
                mean[1][i] *= scale;
            }
        }
        //! The attention output in place of the queries.
        [unroll]
        for (int column = 0; column < 2; ++column)
            mean[column].Store<rowMajor>(memory, queryAt + int(subgroup) * 16 * headStride + column * 16, headStride);
        GroupMemoryBarrierWithGroupSync();
        if (stage == stageAttention) {
            store(origin, thread, queryAt, headStride, headSize, heads * headSize / 8, head * headSize / 8);
            return;
        }

        //! The part of the projection that reads this head.
        [unroll]
        for (int k = 0; k < headSize; k += 16) {
            MatrixA a[rowsPerSubgroup];
            [unroll]
            for (int i = 0; i < rowsPerSubgroup; ++i)
                a[i] = MatrixA.Load<rowMajor>(memory, queryAt + ownedRow(subgroup, i) * 16 * headStride + k, headStride);
            [unroll]
            for (int j = 0; j < colsPerSubgroup; ++j) {
                int column = ownedColumn(subgroup, j);
                if (column >= tilesC)
                    continue;
                MatrixB b = MatrixB.Load<columnMajor>(matrices, tileAt(constants.projection, column * 16, head * headSize + k, attentionChannels), 16);
                [unroll]
                for (int i = 0; i < rowsPerSubgroup; ++i)
                    sums[i][j] = linalg.coopMatMulAdd<half, false>(a[i], b, sums[i][j]);
            }
        }
    }

    //! x + attentionScale · (projection + bias) in place of the block input; then the sums start from the bias of the second MLP layer.
    [unroll]
    for (int j = 0; j < colsPerSubgroup; ++j) {
        int column = ownedColumn(subgroup, j);
        if (column >= tilesC)
            continue;
        [unroll]
        for (int i = 0; i < rowsPerSubgroup; ++i) {
            int at = ownedRow(subgroup, i) * 16 * xStride + column * 16;
            Product x = Product.Load<rowMajor>(memory, at, xStride);
            for (int e = 0; e < x.getCount(); ++e)
                x[e] = x[e] + half(constants.attentionScale) * sums[i][j][e];
            x.Store<rowMajor>(memory, at, xStride);
            sums[i][j] = loadBias(constants.mlpBias + column * 16);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    //! The MLP, 64 hidden channels at a time: the subgroup computes GELU(x W1 + b1) for one column tile of all tokens, then its part of the
    //! second layer.
    for (int chunk = 0; chunk < 4 * channels; chunk += 64) {
        {
            uint hiddenColumn = uint(chunk) + subgroup * 16;
            Product hidden[4];
            [unroll]
            for (int r = 0; r < 4; ++r)
                hidden[r] = loadBias(constants.hiddenBias + hiddenColumn);
            [unroll]
            for (int k = 0; k < channels; k += 16) {
                MatrixB b = MatrixB.Load<columnMajor>(matrices, tileAt(constants.hidden, int(hiddenColumn), k, channels), 16);
                [unroll]
                for (int r = 0; r < 4; ++r)
                    hidden[r] = linalg.coopMatMulAdd<half, false>(MatrixA.Load<rowMajor>(memory, r * 16 * xStride + k, xStride), b, hidden[r]);
            }
            [unroll]
            for (int r = 0; r < 4; ++r) {
                for (int e = 0; e < hidden[r].getCount(); ++e)
                    hidden[r][e] = gelu(hidden[r][e]);
                hidden[r].Store<rowMajor>(memory, queryAt + r * 16 * wideStride + subgroup * 16, wideStride);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (int k = 0; k < 64; k += 16) {
            MatrixA a[rowsPerSubgroup];
            [unroll]
            for (int i = 0; i < rowsPerSubgroup; ++i)
                a[i] = MatrixA.Load<rowMajor>(memory, queryAt + ownedRow(subgroup, i) * 16 * wideStride + k, wideStride);
            [unroll]
            for (int j = 0; j < colsPerSubgroup; ++j) {
                int column = ownedColumn(subgroup, j);
                if (column >= tilesC)
                    continue;
                MatrixB b = MatrixB.Load<columnMajor>(matrices, tileAt(constants.mlp, column * 16, chunk + k, 4 * channels), 16);
                [unroll]
                for (int i = 0; i < rowsPerSubgroup; ++i)
                    sums[i][j] = linalg.coopMatMulAdd<half, false>(a[i], b, sums[i][j]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    //! x + mlpScale · (MLP + bias).
    [unroll]
    for (int j = 0; j < colsPerSubgroup; ++j) {
        int column = ownedColumn(subgroup, j);
        if (column >= tilesC)
            continue;
        [unroll]
        for (int i = 0; i < rowsPerSubgroup; ++i) {
            int at = ownedRow(subgroup, i) * 16 * xStride + column * 16;
            Product x = Product.Load<rowMajor>(memory, at, xStride);
            for (int e = 0; e < x.getCount(); ++e)
                x[e] = x[e] + half(constants.mlpScale) * sums[i][j][e];
            x.Store<rowMajor>(memory, at, xStride);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (colourHead != headOutput)
        store(origin, thread, 0, xStride, channels, words, 0);
    //! The down convolution (2x2, stride 2) for the next encoder. A row tile holds the tokens of one 2x2 parity at the 4x4 positions,
    //! which are the 16 coarser pixels of the window: so the convolution is the sum over the row tiles r of x_r W_r. A subgroup computes
    //! the column tiles subgroup, subgroup + 4, ..., through shared memory in place of the queries and values.
    if (nextChannels > 0) {
        int nextStride = nextChannels + 8;
        for (int j = int(subgroup); j < nextChannels / 16; j += 4) {
            Product sum = loadBias(constants.nextBias + j * 16);
            [unroll]
            for (int r = 0; r < 4; ++r)
                [unroll]
                for (int k = 0; k < channels; k += 16)
                    sum = linalg.coopMatMulAdd<half, false>(MatrixA.Load<rowMajor>(memory, r * 16 * xStride + k, xStride),
                                                            MatrixB.Load<columnMajor>(matrices, tileAt(constants.nextWeight, j * 16, r * channels + k, 4 * channels), 16), sum);
            sum.Store<rowMajor>(memory, queryAt + j * 16, nextStride);
        }
        GroupMemoryBarrierWithGroupSync();
        int2 coarseSize = constants.size / 2;
        for (int index = int(thread); index < 16 * nextChannels / 8; index += threads) {
            int i = index / (nextChannels / 8);
            int word = index % (nextChannels / 8);
            int2 q = origin / 2 + int2(i % 4, i / 4);
            if (any(q < 0) || any(q >= coarseSize))
                continue;
            int from = queryAt + i * nextStride + word * 8;
            uint4 bits;
            [unroll]
            for (int e = 0; e < 4; ++e)
                bits[e] = halfBits(float(memory[from + 2 * e])) | halfBits(float(memory[from + 2 * e + 1])) << 16;
            next[(q.y * coarseSize.x + q.x) * (nextChannels / 8) + word] = bits;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (colourHead == headOutput) {
        //! The heads are functions of the first 18 channels: 0 ... 3 the blend (gain, alpha), 4 ... 6 the guide, 7 ... 9
        //! the 2x2 covariance of the post filter, 10 ... 13 the candidate weights (softmax), 14 ... 17 the feedback of the next frame.
        int2 p = tokenPosition(origin, int(thread) % tokens);
        if (any(p < 0) || any(p >= constants.size))
            return;
        float x[18];
        [unroll]
        for (int c = 0; c < 18; ++c)
            x[c] = float(memory[int(thread) % tokens * xStride + c]);
        if (thread < tokens) {
            int pixel = p.y * constants.size.x + p.x;
            float alpha = sigmoid(x[3]);
            float3 up = upColour(coarse, constants.size / 2, p).rgb;
            float3 scaled = up * constants.gain * float3(sigmoid(x[0]), sigmoid(x[1]), sigmoid(x[2]));
            colourOutput[pixel] = half4(min(float4(colour[pixel]) * (1.0f - alpha) + float4(scaled, 0.0f) * alpha, 65504.0f));
            if (all(p < constants.renderSize))
                guide[p] = float4(sigmoid(x[4]), sigmoid(x[5]), sigmoid(x[6]), 0.0f);
            return;
        }
        if (any(p >= constants.renderSize))
            return;
        float sx = sigmaScale * sigmoid(clamp(x[7], -20.0f, 20.0f));
        float sy = sigmaScale * sigmoid(clamp(x[8], -20.0f, 20.0f));
        float correlation = tanh(x[9] * constants.correlationScale);
        shape[p] = float4(sx * sx, sy * sy, correlation * sx * sy, 0.0f);
        float largest = max(max(x[10], x[11]), max(x[12], x[13]));
        float4 exponentials = exp(float4(x[10], x[11], x[12], x[13]) - largest);
        candidateWeights[p] = exponentials / (exponentials.x + exponentials.y + exponentials.z + exponentials.w);
        feedback[p] = tanh(float4(x[14], x[15], x[16], x[17]));
        return;
    }
    //! The colour head. The logits x W^T of the 16 tokens of row tile `subgroup` (W padded to 16 rows) in FP32, through shared memory as
    //! two FP16 parts: in its tile of attention weights and in place of the queries.
    if (colourHead == 0)
        return;
    {
        Sum sum = Sum(0.0f);
        [unroll]
        for (int k = 0; k < channels; k += 16)
            sum = linalg.coopMatMulAdd<float, false>(MatrixA.Load<rowMajor>(memory, int(subgroup) * 16 * xStride + k, xStride),
                                                     MatrixB.Load<columnMajor>(matrices, tileAt(constants.headWeight, 0, k, channels), 16), sum);
        Product high = Product(sum);
        Product low;
        for (int i = 0; i < sum.getCount(); ++i)
            low[i] = half(sum[i] - float(high[i]));
        high.Store<rowMajor>(memory, queryAt + int(subgroup) * 16 * headStride, headStride);
        low.Store<rowMajor>(memory, queryAt + int(subgroup) * 16 * headStride + 16, headStride);
    }
    subgroupSync();
    uint lane = thread % lanes;
    int token = int(subgroup) * 16 + int(lane);
    int2 p = tokenPosition(origin, token);
    if (lane >= 16 || any(p < 0) || any(p >= constants.size))
        return;
    float logits[4];
    [unroll]
    for (int i = 0; i < 4; ++i)
        logits[i] = colourHead == headBlend || i < 3 ? weights[constants.headBias + i] + float(memory[queryAt + int(subgroup) * 16 * headStride + int(lane) * headStride + i])
                                                       + float(memory[queryAt + int(subgroup) * 16 * headStride + int(lane) * headStride + 16 + i])
                                                 : 0.0f;
    int pixel = p.y * constants.size.x + p.x;
    if (colourHead == headBlend) {
        float alpha = sigmoid(logits[3]);
        float3 up = upColour(coarse, constants.size / 2, p).rgb;
        float3 scaled = up * constants.gain * float3(sigmoid(logits[0]), sigmoid(logits[1]), sigmoid(logits[2]));
        float4 blended = float4(colour[pixel]) * (1.0f - alpha) + float4(scaled, 0.0f) * alpha;
        colourOutput[pixel] = half4(min(blended, 65504.0f));
        return;
    }
    float4 c;
    if (colourHead == headGatePooled) {
        int width = 2 * constants.size.x;
        int at = 2 * p.y * width + 2 * p.x;
        c = (float4(colour[at]) + float4(colour[at + 1]) + float4(colour[at + width]) + float4(colour[at + width + 1])) * 0.25f;
    } else
        c = fused ? float4(exposed, 0.0f) : float4(colour[pixel]);
    c = float4(half4(c));
    float3 gated;
    [unroll]
    for (int i = 0; i < 3; ++i)
        gated[i] = min(c[i] * constants.gain * sigmoid(logits[i]), 65504.0f);
    colourOutput[pixel] = half4(half3(gated), 0.0h);
}

[numthreads(threads, 1, 1)]
[shader("compute")]
void block(uint3 group : SV_GroupID, uint thread : SV_GroupIndex)
{
    run<false>(group, thread);
}

[numthreads(threads, 1, 1)]
[shader("compute")]
void stem(uint3 group : SV_GroupID, uint thread : SV_GroupIndex)
{
    run<true>(group, thread);
}
