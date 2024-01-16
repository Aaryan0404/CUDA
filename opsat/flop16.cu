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
hgemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const half* alpha,
     const half* A, int ldA,
     const half* B, int ldB,
     const half* beta,
     half* C, int ldC)
{
  return cublasHgemm(handle, transA, transB,
                      m, n, k,
                      reinterpret_cast<const __half*>(alpha),
                      reinterpret_cast<const __half*>(A), ldA,
                      reinterpret_cast<const __half*>(B), ldB,
                      reinterpret_cast<const __half*>(beta),
                      reinterpret_cast<      __half*>(C), ldC);
}

int M = 5376;
int N = 5376;
int K = 2048;

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
    half *hA = (half *)malloc(M * K * 2);
    half *hB = (half *)malloc(K * N * 2);
    half *hC = (half *)malloc(M * N * 2);
    half *golden = (half *)malloc(M * N * 2);

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            hA[i * K + j] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.0);
        }
        for (int j = 0; j < N; ++j)
        {
            hC[i * N + j] = (half)(0);
            golden[i * N + j] = (half)(0);
        }
    }

    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            hB[n * K + k] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.0);
        }
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    half alpha = 2.0;
    half beta = 3.0;

    half *dA;
    half *dB;
    half *dC;

    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, K * N * 2));
    CUDA_CHECK(cudaMalloc(&dC, M * N * 2));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * 2, cudaMemcpyHostToDevice));

    // warm up
    for (int i = 0; i < 10; ++i)
    {
        hgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    for (int i = 0; i < 200; ++i)
    {
        hgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
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