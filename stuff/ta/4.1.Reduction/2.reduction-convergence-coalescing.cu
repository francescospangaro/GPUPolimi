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

#define DIM       (4096*10000)
#define BLOCK_DIM 256

#define STRIDE_FACTOR 2

// CPU version of the reduction kernel
double reduce_cpu(const double* data, const int length) {
  double sum = 0;
  for (int i = 0; i < length; i++) { sum += data[i]; }
  return sum;
}

// GPU version of the reduction kernel
// Note efficient convergence for threads and more efficient memory access
__global__ void reduce_convergence_gpu(double* __restrict__ input, double* __restrict__ output) {
  const unsigned int i = threadIdx.x;

  // Apply the offset
  input += blockDim.x * blockIdx.x * STRIDE_FACTOR;
  output += blockIdx.x;

  for (unsigned int stride = blockDim.x; stride >= 1; stride /= STRIDE_FACTOR) {
    if (i < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (i == 0) {
    // You could have used only a single memory location and performed an atomicAdd
    *output = input[0];
  }
}

int main() {
  static_assert(DIM % (BLOCK_DIM * STRIDE_FACTOR) == 0, "Input size and block size should be the same");
  std::vector<double> data(DIM);
  for (int i = 0; i < DIM; ++i)
    data[i] = static_cast<double>(rand()) / RAND_MAX; // Random value between 0 and 1 }

  // CPU version
  double sum_cpu = reduce_cpu(data.data(), data.size());

  // GPU version
  double* gpu_data;
  CHECK(cudaMalloc(&gpu_data, DIM * sizeof(double)));
  CHECK(cudaMemcpy(gpu_data, data.data(), DIM * sizeof(double), cudaMemcpyHostToDevice));

  double* gpu_result;
  const dim3 block_dim(BLOCK_DIM, 1, 1);
  const dim3 grid_dim((DIM + (BLOCK_DIM * STRIDE_FACTOR) - 1) / (BLOCK_DIM * STRIDE_FACTOR), 1, 1);
  CHECK(cudaMalloc(&gpu_result, grid_dim.x * sizeof(double)));
  reduce_convergence_gpu<<<grid_dim, block_dim>>>(gpu_data, gpu_result);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  std::vector<double> host_result(grid_dim.x);
  CHECK(cudaMemcpy(host_result.data(), gpu_result, grid_dim.x * sizeof(double), cudaMemcpyDeviceToHost));

  double sum_gpu = 0;
  for (int i = 0; i < grid_dim.x; ++i) { sum_gpu += host_result[i]; }

  if (std::abs(sum_cpu - sum_gpu) > 1e-3) {
    std::cout << "Reduction CPU and GPU are NOT equivalent!" << std::endl;
    std::cout << "CPU: " << sum_cpu << std::endl;
    std::cout << "GPU: " << sum_gpu << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Reduction CPU and GPU are equivalent!" << std::endl;

  // Cleanup
  CHECK(cudaFree(gpu_result));
  CHECK(cudaFree(gpu_data));

  return 0;
}
