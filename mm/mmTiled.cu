#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16  // Assuming TILE_WIDTH is a divisor of the matrix dimension


// Block stats
// Bytes of shared mem = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float)
// Threads per block = TILE_WIDTH * TILE_WIDTH

__global__ void matrixMulTiled(float *A, float *B, float *C, int n) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;
    float sum = 0.0f;

    // Loop over tiles of the input matrices in steps of TILE_WIDTH
    #pragma unroll
    for (int m = 0; m < n / TILE_WIDTH; ++m) {
        
        // Load one tile of A and one tile of B into shared memory
        tileA[ty][tx] = A[row * n + (m*TILE_WIDTH + tx)];
        tileB[ty][tx] = B[(m*TILE_WIDTH + ty) * n + col];
        __syncthreads();

        // Multiply tiles and accumulate result
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
    }

    // Write the result
    C[row * n + col] = sum;
}

// CPU implementation of matrix multiplication - to verify results
void matrixMultiplyCPU(float *A, float *B, float *C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

int main() {
    // Matrix dimensions (assuming square matrices for simplicity)
    int n = 1024;  

    // Size of the matrix (must be a multiple of TILE_WIDTH)
    size_t size = n * n * sizeof(float);

    // Allocate and initialize host matrices
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    srand(time(NULL));
    
    // Clarify if we need to modify kernel to be able 
    // to handle double precision

    // Initialize matrices with some values
    for (int i = 0; i < n * n; ++i) {
        h_A[i] = static_cast<float>(rand() % 16); // Generates rand int [0, 15] - then converts to float
        h_B[i] = static_cast<float>(rand() % 16); // Same as above
    }

    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(n / TILE_WIDTH, n / TILE_WIDTH);

    // Launch kernel
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    // Copy result matrix back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float *h_C_CPU = (float *)malloc(size);
    matrixMultiplyCPU(h_A, h_B, h_C_CPU, n);

    // Compare the results
    for (int i = 0; i < n * n; i++) {
        if (fabs(h_C_CPU[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    // Free CPU result memory
    free(h_C_CPU);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
