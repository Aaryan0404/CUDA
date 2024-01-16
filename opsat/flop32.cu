// nvcc -arch=sm_89 -lcublas flop.cu -o test_cublas

// ./test_cublas M 5376 N 5376 K 2048

// Test performance using shape M=10240, N=10240, K=4096
// Running cost (ms) of CuBLAS is 5.15534
// TFLOPS: 166.622

#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <string>
#include <cassert>
#include <cublas_v2.h>

// hgemm
inline cublasStatus_t
sgemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const float* alpha,
     const float* A, int ldA,
     const float* B, int ldB,
     const float* beta,
     float* C, int ldC)
{
  return cublasSgemm(handle, transA, transB,
                      m, n, k,
                      reinterpret_cast<const float*>(alpha),
                      reinterpret_cast<const float*>(A), ldA,
                      reinterpret_cast<const float*>(B), ldB,
                      reinterpret_cast<const float*>(beta),
                      reinterpret_cast<      float*>(C), ldC);
}

int M = 512;
int N = 512;
int K = 512;

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

int main(int argc, char *argv[])
{
  if (argc > 1)
  {
      assert((argc - 1) % 2 == 0);
      for (int i = 1; i < argc; i += 2)
      {
          char *key = argv[i];
          char *value = argv[i + 1];
          std::string keys(key);
          if (keys == "M") {
              M = std::atoi(value);
          } else if (keys == "N") {
              N = std::atoi(value);
          } else if (keys == "K") {
              K = std::atoi(value);
          }
      }
  }

    std::cout << "Test performance using shape M=" << M << ", N=" << N << ", K=" << K << "\n";
    srand(time(NULL));
    float *hA = (float *)malloc(M * K * 4);
    float *hB = (float *)malloc(K * N * 4);
    float *hC = (float *)malloc(M * N * 4);
    float *golden = (float *)malloc(M * N * 4);

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            hA[i * K + j] = (float)(rand() % 1000 * 1 / 100 % 10 + 0.0);
        }
        for (int j = 0; j < N; ++j)
        {
            hC[i * N + j] = (float)(0);
            golden[i * N + j] = (float)(0);
        }
    }

    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            hB[n * K + k] = (float)(rand() % 1000 * 1 / 100 % 10 + 0.0);
        }
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float alpha = 2.0;
    float beta = 3.0;

    float *dA;
    float *dB;
    float *dC;

    CUDA_CHECK(cudaMalloc(&dA, M * K * 4));
    CUDA_CHECK(cudaMalloc(&dB, K * N * 4));
    CUDA_CHECK(cudaMalloc(&dC, M * N * 4));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * 4, cudaMemcpyHostToDevice));

    // warm up
    for (int i = 0; i < 10; ++i)
    {
        sgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    for (int i = 0; i < 200; ++i)
    {
        sgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Running cost (ms) of CuBLAS is " << ms / 200.0 << "\n";
    std::cout << "TFLOPS: " << ((2.0f * M * N * K) + (M * N)) / (ms / 200.0 / 1000.0) / 1e12 << "\n";

    free(hA);
    free(hB);
    free(hC);
    free(golden);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}