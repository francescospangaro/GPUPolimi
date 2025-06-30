/*
 * An example of using shared memory to transpose square thread coordinates
 * of a CUDA grid into a global memory array. Different kernels below
 * demonstrate performing reads and writes with different ordering, as well as
 * optimizing using memory padding.
 * Good example here http://cuda-programming.blogspot.com/2013/02/bank-conflicts-in-shared-memory-in-cuda.html
 */

// NOTE warp are organized starting from threadIdx.x which is the fastest value changing

#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <iostream>

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

#define BDIMX 32
#define BDIMY 32
#define IPAD 1

void printData(std::string msg, int *in, const int size)
{
  std::cout << msg << ": ";

  for (int i = 0; i < size; i++)
  {
    printf("%5d", in[i]);
    fflush(stdout);
  }

  printf("\n");
  return;
}

__global__ void setRowReadRow(int *__restrict__ out)
{
  // static shared memory
  __shared__ int tile[BDIMY][BDIMX];

  // mapping from thread index to global memory index
  const unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

  // shared memory store operation
  tile[threadIdx.y][threadIdx.x] = idx;

  // wait for all threads to complete
  __syncthreads();

  // shared memory load operation
  out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *__restrict__ out)
{
  // static shared memory
  __shared__ int tile[BDIMX][BDIMY];

  // mapping from thread index to global memory index
  const unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

  // shared memory store operation
  tile[threadIdx.x][threadIdx.y] = idx;

  // wait for all threads to complete
  __syncthreads();

  // shared memory load operation
  out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *__restrict__ out)
{
  // static shared memory
  __shared__ int tile[BDIMY][BDIMX];

  // mapping from thread index to global memory index
  const unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

  // shared memory store operation
  tile[threadIdx.y][threadIdx.x] = idx;

  // wait for all threads to complete
  __syncthreads();

  // shared memory load operation
  out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColPad(int *__restrict__ out)
{
  // static shared memory
  __shared__ int tile[BDIMY][BDIMX + IPAD];

  // mapping from thread index to global memory offset
  const unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

  // shared memory store operation
  tile[threadIdx.y][threadIdx.x] = idx;

  // wait for all threads to complete
  __syncthreads();

  // shared memory load operation
  out[idx] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char **argv)
{
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("%s at ", argv[0]);
  printf("device %d: %s ", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  // Not supported by zluda
  // cudaSharedMemConfig pConfig;
  // CHECK(cudaDeviceGetSharedMemConfig ( &pConfig ));
  // printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");

  // set up array size 2048
  int nx = BDIMX;
  int ny = BDIMY;

  bool iprintf = 0;

  if (argc > 1)
    iprintf = atoi(argv[1]);

  size_t nBytes = nx * ny * sizeof(int);

  // execution configuration
  dim3 block(BDIMX, BDIMY);
  dim3 grid(1, 1);
  printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
         block.y);

  // allocate device memory
  int *d_C;
  CHECK(cudaMalloc((int **)&d_C, nBytes));
  int *gpuRef = (int *)malloc(nBytes);

  CHECK(cudaMemset(d_C, 0, nBytes));
  setColReadCol<<<grid, block>>>(d_C);
  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

  if (iprintf)
    printData("set col read col   ", gpuRef, nx * ny);

  CHECK(cudaMemset(d_C, 0, nBytes));
  setRowReadRow<<<grid, block>>>(d_C);
  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

  if (iprintf)
    printData("set row read row   ", gpuRef, nx * ny);

  CHECK(cudaMemset(d_C, 0, nBytes));
  setRowReadCol<<<grid, block>>>(d_C);
  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

  if (iprintf)
    printData("set row read col   ", gpuRef, nx * ny);

  CHECK(cudaMemset(d_C, 0, nBytes));
  setRowReadColPad<<<grid, block>>>(d_C);
  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

  if (iprintf)
    printData("set row read col pad", gpuRef, nx * ny);

  // free host and device memory
  CHECK(cudaFree(d_C));
  free(gpuRef);

  // reset device
  CHECK(cudaDeviceReset());
  return EXIT_SUCCESS;
}
