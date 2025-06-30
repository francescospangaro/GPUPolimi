#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define DIM           (1024)
#define BLOCK_DIM     DIM
#define STRIDE_FACTOR 2

// CPU version of the scan kernel EXCLUSIVE
void scan_cpu(const float* input, float* output, const int length) {
  output[0] = 0;
  for (int i = 1; i < length; ++i) { output[i] = output[i - 1] + input[i - 1]; }
}

// GPU version of the scan kernel EXCLUSIVE
__global__ void
    kogge_stone_scan_gpu(const float* __restrict__ input, float* __restrict__ output, const int length) {
  __shared__ float input_s[BLOCK_DIM];
  const unsigned int tid = threadIdx.x;

  input_s[tid] = input[tid - 1];
  for (unsigned int stride = 1; stride < length; stride *= STRIDE_FACTOR) {
    __syncthreads();
    float temp;
    if (tid >= stride)
      temp = input_s[tid] + input_s[tid - stride];
    __syncthreads();
    if (tid >= stride)
      input_s[tid] = temp;
      // input_s[tid] = input_s[tid] + input_s[tid - stride];
  }
  output[tid] = input_s[tid];

  return;
}

int main() {
  std::vector<float> input(DIM);
  std::vector<float> output(DIM);
  for (int i = 0; i < DIM; ++i)
    input[i] = static_cast<float>(rand()) / RAND_MAX; // Random value between 0 and 1 }

  // CPU version
  scan_cpu(input.data(), output.data(), input.size());

  // GPU version
  float *input_d, *output_d;
  CHECK(cudaMalloc(&input_d, DIM * sizeof(float)));
  CHECK(cudaMalloc(&output_d, DIM * sizeof(float)));
  CHECK(cudaMemcpy(input_d, input.data(), DIM * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 block_dim(BLOCK_DIM, 1, 1);
  const dim3 grid_dim(1, 1, 1);
  kogge_stone_scan_gpu<<<grid_dim, block_dim>>>(input_d, output_d, DIM);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  std::vector<float> host_result(DIM);
  CHECK(cudaMemcpy(host_result.data(), output_d, DIM * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < host_result.size(); ++i) {
    if (std::abs(output[i] - host_result[i]) > 1e-3) {
      std::cout << "Scan CPU and GPU are NOT equivalent!" << std::endl;
      std::cout << "Index: " << i << std::endl;
      std::cout << "CPU: " << output[i] << std::endl;
      std::cout << "GPU: " << host_result[i] << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << "Scan CPU and GPU are equivalent!" << std::endl;

  // Cleanup
  CHECK(cudaFree(output_d));
  CHECK(cudaFree(input_d));

  return 0;
}
