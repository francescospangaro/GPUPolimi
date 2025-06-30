#include <stdio.h>
#include <stdlib.h>
// #include <sys/time.h>
#include <time.h>

#include <iostream>
#include <fstream>
#include <string>

#define MAX_LENGTH 50000000

#define WARPSIZE 32
// For shuffle block dim need to have the same dimension of a warp
#define BLOCKDIM WARPSIZE
#define GRIDDIM 2048

#define CHAR_PER_BIN 6
#define ALPHABET_SIZE 26
#define BIN_NUM ((ALPHABET_SIZE - 1) / CHAR_PER_BIN + 1)
#define FIRST_CHAR 'a'

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess)                                                         \
    {                                                                               \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                         \
    {                                                                               \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

/*
double get_time() // function to get the time of day in seconds
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
*/

void sequential_histogram(const char *data, unsigned int *histogram, const int length)
{
  for (int i = 0; i < length; i++)
  {
    int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE) // check if we have an alphabet char
      histogram[alphabet_position / CHAR_PER_BIN]++;                 // we group the letters into blocks of 6
  }
}

__global__ void histogram_kernel(const char *__restrict__ data,
                                 unsigned int *__restrict__ histogram,
                                 const unsigned int length)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  // Privatized bins
  __shared__ unsigned int histo_s[BIN_NUM];
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x)
    histo_s[binIdx] = 0u;
  __syncthreads();
  // Histogram
  for (unsigned int i = tid; i < length; i += stride)
  {
    const int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
      atomicAdd(&(histo_s[alphabet_position / CHAR_PER_BIN]), 1);
  }
  __syncthreads();
// Commit to global memory
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x)
  {
    const unsigned int binValue = histo_s[binIdx];
    if (binValue > 0)
    {
      atomicAdd(&(histogram[binIdx]), binValue);
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("Please provide a filename as an argument.\n");
    return 1;
  }

  const char *filename = argv[1];
  std::fstream fp(filename);

  std::string txt;
  while (std::getline(fp, txt))
  {
    printf("Retrieved line of length %zd:\n", txt.length());
  }
  fp.close();

  const char *text = txt.c_str();
  int len = txt.length();
  char *text_d;

  unsigned int histogram[BIN_NUM] = {0};
  unsigned int histogram_hw[BIN_NUM] = {0};
  unsigned int *histogram_d;
  sequential_histogram(text, histogram, len);

  CHECK(cudaMalloc(&text_d, len * sizeof(char)));                              // allocate space for the input array on the GPU
  CHECK(cudaMalloc(&histogram_d, BIN_NUM * sizeof(unsigned int)));             // and for the histogram
  CHECK(cudaMemcpy(text_d, text, len * sizeof(char), cudaMemcpyHostToDevice)); // copy input data on the gpu

  dim3 blocksPerGrid(GRIDDIM, 1, 1);
  dim3 threadsPerBlock(BLOCKDIM, 1, 1);
  histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(text_d,
                                                       histogram_d,
                                                       len); // same nof as the regular histogram
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(histogram_hw,
                   histogram_d,
                   BIN_NUM * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost)); // copy data back from the gpu
  for (size_t i = 0; i < BIN_NUM; i++)
  {
    if (histogram[i] != histogram_hw[i])
    {
      printf("Error on GPU at index: %zd\n", i);
      return 0;
    }
  }
  printf("ALL GPU OK\n");

  CHECK(cudaFree(text_d));
  CHECK(cudaFree(histogram_d));

  return 1;
}
