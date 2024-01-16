#include "common.h"

#define MMA_M 16               // # of rows in a tile 
#define MMA_N 8                // # of cols in a tile
#define MMA_K 16               // # of cols in A and rows in B in a tile

// A = (M rows, K cols)
// B = (K rows, N cols)
// C = (M rows, N cols)

#define BLOCK_ROWS 256         // # of rows to be processed per block
#define BLOCK_COLS 128         // # of cols to be processed per block

#define WARP_ROWS 64           // # of rows to be processed per warp
#define WARP_COLS 64           // # of cols to be processed per warp

#define BLOCK_ROW_WARPS 2      // # of warps in a block row == BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4      // # of warps in a block col == BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 16     // # of row_tiles in a block == BLOCK_ROWS / MMA_M
#define BLOCK_COL_TILES 16     // # of col_tiles in a block == BLOCK_COLS / MMA_N

#define WARP_ROW_TILES 8       // # of row_tiles in a warp == WARP_COLS / MMA_N
#define WARP_COL_TILES 4       // # of col_tiles in a warp == WARP_ROWS / MMA_M

#define WARP_SIZE 32           // # of threads in a warp
#define WARPS_PER_BLOCK 8      // # of warps in a block   == BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // # of threads per block  == WARP_SIZE * WARPS_PER_BLOCK

                                     // https://stackoverflow.com/questions/10833953/cuda-global-memory-copy
                                     // Idea = whenever data is accessed a block of 32/64/128-bytes has to be read
                                     // cache line size = 32/64/128 bytes
                                     // hypothesis: 32-element line == max cache line size of 128 bytes / sizeof(half)
#define CHUNK_K 2                    // # of K_tiles per 32-element line of K = 32 / MMA_K    (32-element line of K = line_K)

                                     // https://stackoverflow.com/questions/10833953/cuda-global-memory-copy 
                                     // Set-up for how you can get each thread fo access 16 bytes of mem at once
#define THREAD_COPY_BYTES 16         // # of bytes one thread should copy

#define CHUNK_LINE_BYTES 64          // # of bytes to be copied per warp == (num-elements in line_K) * element-size = CHUNK_K * MMA_K * sizeof(half)

#define CHUNK_COPY_LINES_PER_WARP 8  // # of line_Ks to be copied per warp  == (WARP_SIZE * THREAD_COPY_BYTES) / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // # of threads per line_K             == WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 32  // num-elements in line_K = CHUNK_K * MMA_K

#define C_SMEM_STRIDE 128  // # of cols to be processed per block = BLOCK_COLS
#define C_SMEM_OFFSET 64   // # of cols to be processed per warp  = WARP_COLS

#define BLOCK_STRIDE 16    // x dim of grid - let's call this # of blocks in a mega-block

#define SMEM_BANK_ROWS 2   // (cache line size) / (line_k_size * sizeof_line_k_element) = (32 elements * 4 bytes/element) / (AB_SMEM_STRIDE * sizeof(half)   

#define PERMUTED_OFFSET 8 
#define PERMUTED_COLS 4

#define K_STAGE 4

__device__ void warp_mma(size_t reg_load_idx, uint32_t (&RC)[WARP_COL_TILES][WARP_ROW_TILES][2],
                         uint32_t (&RA)[2][WARP_COL_TILES][4], uint32_t (&RB)[2][WARP_ROW_TILES][2]) {
    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        // for loop across warp_row_tiles
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

            // hmma16816 - half precision matrix multiply accumulate
            // use load_registers to supply data to the hmma instruction
            HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                        RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                        RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
        }
    }
}

__device__ void g_to_s_A(size_t A_smem_iters, size_t A_smem_idx, size_t lane_id, int4 *A_lane_ptr, size_t K, half smem[][AB_SMEM_STRIDE]) {
    // parallely copy data from global memory to shared memory (matrix A)
    #pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        // cp_async_cg - copy async with commit group
        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }
}

__device__ void g_to_s_B(size_t B_smem_iters, size_t B_smem_idx, size_t lane_id, int4 *B_lane_ptr, size_t K, half smem[][AB_SMEM_STRIDE]) {
    // parallely copy data from global memory to shared memory (matrix B)
    #pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        // cp_async_cg - copy async with commit group
        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }
}

__device__ void ld_x4(size_t smem_load_off, size_t warp_id, size_t lane_id, uint32_t (&RA)[2][WARP_COL_TILES][4], half smem[][AB_SMEM_STRIDE], size_t reg_store_idx) {
    // parallely copy data from shared memory to registers (matrix A)
    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
            &smem[A_smem_idx + lane_id % 16][((lane_id / 16) * 8 + (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) /
                                                                       SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                                             AB_SMEM_STRIDE]);

        // ldmatrix_x4 - store 4 elements from shared memory to registers
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    A_smem_lane_addr);
    }
}

__device__ void ld_x2(size_t smem_load_off, size_t warp_id, size_t lane_id, uint32_t (&RB)[2][WARP_ROW_TILES][2], half smem[][AB_SMEM_STRIDE], size_t reg_store_idx,
                      size_t B_smem_idx_off) {
    // parallely copy data from shared memory to registers (matrix B)
    #pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
        size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
        uint32_t B_smem_lane_addr =
            __cvta_generic_to_shared(&smem[B_smem_idx + lane_id % 8]
                                          [(((lane_id / 8) % 2) * 8 + (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) /
                                                                          SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                                           AB_SMEM_STRIDE]);

        // ldmatrix_x2 - store 2 elements from shared memory to registers
        LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
    }
}

__global__ void mma_myk(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                     size_t M, size_t N, size_t K) {

    // number of tiles in each dimension
    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t N_tiles = div_ceil(N, MMA_N);
    const size_t K_tiles = div_ceil(K, MMA_K);

    // now, each thread calculates the i, j indices of the tile it is responsible for

    // calculate row tile index in C
    //   if mega col block idx even -> row block idx goes from gridDim.y - 1 to 0
    //   if mega col block idx odd  -> row block idx goes from 0 to gridDim.y - 1
    const size_t block_tile_i = (blockIdx.z % 2) ? (((gridDim.y - 1) - blockIdx.y) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);

    // calculate col tile index in C
    //   blockIdx.z * gridDim.x + blockIdx.x = col block idx
    //   so, col tile idx = col block idx * BLOCK_ROW_TILES
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    // if the tile is out of bounds, return
    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    // shared memory allocation
    extern __shared__ half smem[][AB_SMEM_STRIDE];

    // warp_id = same for 32 threads in a warp
    const size_t warp_id = threadIdx.x / WARP_SIZE;

    // lane_id = identifier for each thread in a warp
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    // B_smem_idx_off - offset for B in shared memory
    constexpr size_t B_smem_idx_off = BLOCK_ROWS;

    // smem_stage_off - offset for each stage in shared memory
    constexpr size_t smem_stage_off = BLOCK_ROWS + BLOCK_COLS;

    // smem_warp_tile_row_ptr = pointer to the start of the row of tiles in shared memory
    half *smem_warp_tile_row_ptr = &smem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS;

    // smem_warp_tile_col_ptr = pointer to the start of the col of tiles in shared memory
    const half *smem_warp_stream_ptr = &smem[0][0] + warp_id * MMA_M * 2 * C_SMEM_STRIDE;

    // gmem_idx - location to write to in C in global memory
    const size_t gmem_idx = (block_tile_i + warp_id * 2) * MMA_M * N + block_tile_j * MMA_N;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // register allocation - this behaves like a register tile, which is 
    // shared memory between (32) threads in a warp
    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2];

    // parallely initialize all the registers to 0
    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    // get ptrs to the start of the row and col tiles in shared memory
    // these are the ptrs that will be used to load the data from global memory
    const half *A_warp_ptr = &A[block_tile_i * MMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const half *B_warp_ptr = &B[block_tile_j * MMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    // num of itrs to copy data from global memory to shared memory
    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

    size_t smem_store_idx = 0;
    size_t smem_load_idx = 0;

    size_t smem_store_off = 0;
    size_t smem_load_off = 0;

    size_t A_smem_idx = 0;
    int4 *A_lane_ptr = nullptr;

    size_t B_smem_idx = 0;
    int4 *B_lane_ptr = nullptr;

    // STAGE 0 (smem store)

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

    g_to_s_A(A_smem_iters, A_smem_idx, lane_id, A_lane_ptr, K, smem);

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;
    
    g_to_s_B(B_smem_iters, B_smem_idx, lane_id, B_lane_ptr, K, smem);

    // commit group - commit all the async copies in the group
    CP_ASYNC_COMMIT_GROUP();


    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;
    // STAGE 1 (smem store)

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

    g_to_s_A(A_smem_iters, A_smem_idx, lane_id, A_lane_ptr, K, smem);

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

    g_to_s_B(B_smem_iters, B_smem_idx, lane_id, B_lane_ptr, K, smem);

    // commit group - commit all the async copies in the group
    CP_ASYNC_COMMIT_GROUP();


    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;
    // STAGE 2 (smem store)

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + 2 * CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

    g_to_s_A(A_smem_iters, A_smem_idx, lane_id, A_lane_ptr, K, smem);

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + 2 * CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

    g_to_s_B(B_smem_iters, B_smem_idx, lane_id, B_lane_ptr, K, smem);

    // commit group - commit all the async copies in the group
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(2);

    __syncthreads();


    // Now, we have to load the data from shared memory to registers

    uint32_t RA[2][WARP_COL_TILES][4];
    uint32_t RB[2][WARP_ROW_TILES][2];

    // store into RA[0] and RB[0]
    size_t reg_store_idx = 0;
    // load from RA[1] and RB[1]
    size_t reg_load_idx = 1;

    // STAGE 0 (smem load)
    ld_x4(smem_load_off, warp_id, lane_id, RA, smem, reg_store_idx);
    ld_x2(smem_load_off, warp_id, lane_id, RB, smem, reg_store_idx, B_smem_idx_off);

    // finally, we have to do the actual computation by looping across K_tiles (shared dim of A and B)
    #pragma unroll
    for (size_t tile_k = CHUNK_K * (K_STAGE - 1); tile_k < K_tiles; tile_k += CHUNK_K) {

        // we will store where we loaded from
        reg_store_idx ^= 1;
        // we will load where we stored
        reg_load_idx ^= 1;

        // load data from shared memory to registers (PART of SMEM LOAD - stage defined in line 301 for tile_k = 0, or line 436 otherwise)
        ld_x4(smem_load_off, warp_id, lane_id, RA, smem, reg_store_idx);

        // load data from shared memory to registers (PART of SMEM LOAD - stage define in line 301 for tile_k = 0) or line 436 otherwise)
        ld_x2(smem_load_off, warp_id, lane_id, RB, smem, reg_store_idx, B_smem_idx_off);

        // meanwhile above two (loading into registers) for loops, actual computation below
        // warp_mma - half precision matrix multiply accumulate
        warp_mma(reg_load_idx, RC, RA, RB);

        smem_store_idx = (smem_store_idx + 1) % K_STAGE;
        smem_store_off = smem_store_idx * smem_stage_off;
        // STAGE 3, 0, 1, 2, 3, 0, 1, 2, 3, ... (smem store)

        A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

        // parallely copy data from global memory to shared memory (matrix A)
        #pragma unroll
        for (size_t i = 0; i < A_smem_iters / CHUNK_K; ++i) {
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;
                
            // cp_async_cg - copy async with commit group
            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

        // parallely copy data from global memory to shared memory (matrix B)
        #pragma unroll
        for (size_t i = 0; i < B_smem_iters / CHUNK_K; ++i) {
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;
            
            // cp_async_cg - copy async with commit group
            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        smem_load_idx = (smem_load_idx + 1) % K_STAGE;
        smem_load_off = smem_load_idx * smem_stage_off;
        // STAGE 1, 2, 3, 0, 1, 2, 3, ... (smem load)
        // to see - skip to line 479

        // parallely copy data from global memory to shared memory (matrix A) (PART of SMEM STORE - stage defined in line 388)
        #pragma unroll
        for (size_t i = (CHUNK_K - 1) * A_smem_iters / CHUNK_K; i < A_smem_iters; ++i) {
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;
            
            // cp_async_cg - copy async with commit group
            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        // parallely copy data from global memory to shared memory (matrix B) (PART of SMEM STORE - stage defined in line 388)
        #pragma unroll
        for (size_t i = (CHUNK_K - 1) * B_smem_iters / CHUNK_K; i < B_smem_iters; ++i) {
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;
            
            // cp_async_cg - copy async with commit group
            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        // commit group - commit all the async copies in the group
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(2);

        __syncthreads();

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        // parallely copy data from shared memory to registers (matrix A) (PART of SMEM LOAD - stage defined in line 434)
        ld_x4(smem_load_off, warp_id, lane_id, RA, smem, reg_store_idx);

        // parallely copy data from shared memory to registers (matrix B) (PART of SMEM LOAD - stage defined in line 434)
        ld_x2(smem_load_off, warp_id, lane_id, RB, smem, reg_store_idx, B_smem_idx_off);

        // warp_mma - half precision matrix multiply accumulate
        warp_mma(reg_load_idx, RC, RA, RB);
    }

    // finish the computation for remaining K_tiles (same behavior as k_tile for loop starting
    //                                               from line 333)                               
    #pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        // parallely copy data from shared memory to registers (matrix A) (PART of SMEM LOAD)
        ld_x4(smem_load_off, warp_id, lane_id, RA, smem, reg_store_idx);

        // parallely copy data from shared memory to registers (matrix B) (PART of SMEM LOAD)
        ld_x2(smem_load_off, warp_id, lane_id, RB, smem, reg_store_idx, B_smem_idx_off);

        // warp_mma - half precision matrix multiply accumulate
        warp_mma(reg_load_idx, RC, RA, RB);

        if (k_step + 2 == CHUNK_K) {
            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;
            // STAGE 0, 1, 2, 3, ... (smem load)

            CP_ASYNC_WAIT_GROUP(1);

            __syncthreads();
        }
    }

    // identical to for loop starting on line 529
    #pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        // parallely copy data from shared memory to registers (matrix A) (PART of SMEM LOAD)
        ld_x4(smem_load_off, warp_id, lane_id, RA, smem, reg_store_idx);

        // parallely copy data from shared memory to registers (matrix B) (PART of SMEM LOAD)
        ld_x2(smem_load_off, warp_id, lane_id, RB, smem, reg_store_idx, B_smem_idx_off);

        // warp_mma - half precision matrix multiply accumulate
        warp_mma(reg_load_idx, RC, RA, RB);

        if (k_step + 2 == CHUNK_K) {
            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;
            // STAGE 0, 1, 2, 3, ... (smem load)

            CP_ASYNC_WAIT_GROUP(0);

            __syncthreads();
        }
    }

    // identical to for loop starting on line 529, 591 (except k_step = 1 instead of 0)
    #pragma unroll
    for (size_t k_step = 1; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        // parallely copy data from shared memory to registers (matrix A) (PART of SMEM LOAD)
        ld_x4(smem_load_off, warp_id, lane_id, RA, smem, reg_store_idx);

        // parallely copy data from shared memory to registers (matrix B) (PART of SMEM LOAD)
        ld_x2(smem_load_off, warp_id, lane_id, RB, smem, reg_store_idx, B_smem_idx_off);

        // warp_mma - half precision matrix multiply accumulate
        warp_mma(reg_load_idx, RC, RA, RB);
    }

    // FINAL COMPUTATION
    // warp_mma - half precision matrix multiply accumulate
    warp_mma(reg_store_idx, RC, RA, RB);

    __syncthreads();

    // store the data from registers to shared memory
    // for loop across warp_col_tiles
    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        // for loop across warp_row_tiles
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            half *lane_ptr0 =
                smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4) * C_SMEM_STRIDE +
                ((warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET + j * MMA_N +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4) % 8) * PERMUTED_OFFSET) %
                    C_SMEM_STRIDE;
            half *lane_ptr1 =
                smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4 + 8) * C_SMEM_STRIDE +
                ((warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET + j * MMA_N +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4 + 8) % 8) * PERMUTED_OFFSET) %
                    C_SMEM_STRIDE;
            
            // stmatrix_x2 - store 2 elements from registers to shared memory
            *((uint32_t *)(lane_ptr0)) = RC[i][j][0];
            *((uint32_t *)(lane_ptr1)) = RC[i][j][1];
        }
    }

    __syncthreads();

    // finally, store the data from shared memory to global memory
    #pragma unroll
    for (size_t i = 0; i < MMA_M; ++i) {
        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) +
              (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % (C_SMEM_STRIDE * sizeof(half) / THREAD_COPY_BYTES));
    }
}

size_t initmyKernel() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half) * K_STAGE,
                                    BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(cudaFuncSetAttribute(mma_myk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void mma_myKernel(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initmyKernel();

    dim3 block(THREADS_PER_BLOCK);

    // grid.x - number of blocks in a mega block
    // grid.y - number of blocks in a grid column   (total number of rows / number of rows in a block)
    // grid.z - number of mega blocks in a grid row (total number of cols / number of cols in a block / number of blocks in a mega block)
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    mma_myk<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}
