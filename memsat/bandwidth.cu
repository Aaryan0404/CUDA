#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void memBoundKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

int main() {
    for (int i = 20; i <= 32; i ++) {

        int dataSize = (pow(2, i))/sizeof(float);
        size_t bytes = dataSize * sizeof(float);

        // print data size
        printf("Data size: %d\n", dataSize);

        float* h_input = (float*)malloc(bytes);
        float* h_output = (float*)malloc(bytes);

        for (int i = 0; i < dataSize; i++) {
            h_input[i] = static_cast<float>(i);
        }

        float *d_input, *d_output;
        cudaMalloc((void**)&d_input, bytes);
        cudaMalloc((void**)&d_output, bytes);

        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (dataSize + blockSize - 1) / blockSize;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        memBoundKernel<<<numBlocks, blockSize>>>(d_input, d_output, dataSize);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

        for (int i = 0; i < dataSize; i++) {
            if (h_output[i] != h_input[i]) {
                fprintf(stderr, "Verification failed at index %d!\n", i);
                return -1;
            }
        }

        printf("\n");
        printf("Verification successful!\n");
        printf("Kernel execution time: %f ms\n", milliseconds);
        float bandwidth = (2.0f * (float)dataSize * sizeof(float)) / (milliseconds * 1000000.0f);
        printf("Calculated memory bandwidth: %f GB/s\n", bandwidth);

        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);

    }

    return 0;
}