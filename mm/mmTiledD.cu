#include <cuda_runtime.h>
#include <stdio.h>

// Modify TILE_WIDTH dynamically
#define MAX_TILE_WIDTH 32  // Maximum TILE_WIDTH you want to test
    // This is 32, because we cannot have more than 32 x 32 threads in a 
    // thread block (although we still have room in shared mem)

// Block stats
// Bytes of shared mem (max would be 48 KB, regularly?) = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float) = 2 * 64 * 64 * 4 = 32.7 KB 
    // IMPORTANT - we run out of threads before we run out of shared mem    
// Threads per block = TILE_WIDTH * TILE_WIDTH
__global__ void matrixMulTiled(float *A, float *B, float *C, int n, int tileWidth) {
    // Use dynamic shared memory based on the tileWidth passed
    extern __shared__ float sharedMemory[];

    float *tileA = sharedMemory;
    float *tileB = &sharedMemory[tileWidth * tileWidth];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * tileWidth + ty;
    int col = blockIdx.x * tileWidth + tx;
    float sum = 0.0f;

    // Loop over tiles of the input matrices in steps of tileWidth
    #pragma unroll
    for (int m = 0; m < n / tileWidth; ++m) {
        // Adjust the indexing for dynamic tile width
        tileA[ty * tileWidth + tx] = A[row * n + (m * tileWidth + tx)];
        tileB[ty * tileWidth + tx] = B[(m * tileWidth + ty) * n + col];
        __syncthreads();

        for (int k = 0; k < tileWidth; ++k) {
            sum += tileA[ty * tileWidth + k] * tileB[k * tileWidth + tx];
        }
        __syncthreads();
    }

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
    int n = 2048;  

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

    // Print cuda device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device Name: %s\n", prop.name);
    printf("Shared Memory Per Block: %lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);


    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    for (int tileWidth = 2; tileWidth <= MAX_TILE_WIDTH; tileWidth *= 2) {
        // Adjust grid and block dimensions based on current tileWidth
        dim3 dimBlock(tileWidth, tileWidth);
        dim3 dimGrid(n / tileWidth, n / tileWidth);

        // Allocate shared memory dynamically
        size_t sharedMemSize = 2 * tileWidth * tileWidth * sizeof(float);

        // Start timer
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Launch kernel with current configuration
        matrixMulTiled<<<dimGrid, dimBlock, sharedMemSize>>>(d_A, d_B, d_C, n, tileWidth);

        // Stop timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Tile Width: %d, Time (ms): %f\n", tileWidth, milliseconds);

        // Cleanup events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Copy result matrix back to host
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        // Verify the result for each configuration
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

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
            // Handle error...
        }

        for (int i = 0; i < n * n; ++i) {
            h_A[i] = static_cast<float>(rand() % 16);
            h_B[i] = static_cast<float>(rand() % 16);
        }
    
        // Copy updated matrices to device
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
