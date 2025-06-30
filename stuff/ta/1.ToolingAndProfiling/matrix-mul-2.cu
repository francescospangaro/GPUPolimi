/*
 * This program executes a matrix multiplication by means of three different implementations:
 * 1) basic algorithm on CPU, 2) basic algorithm on GPU, 3) tiled algorithm on GPU.
 * Compile with:
 * nvcc .\matrix-mul-2.cu -lcublas -o .\matrix-mul-2.exe
 * Analyze the following metrics with nvprof:
 * gld_efficiency,gst_efficiency,gld_transactions,shared_efficiency,shared_load_transactions_per_request,shared_store_transactions_per_request,shared_load_transactions,shared_store_transactions
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

#define BLOCKDIM 32
#define TILE_WIDTH BLOCKDIM
#define MAXVAL 8

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

#define CHECK_CUBLAS(call)                                         \
  {                                                                \
    const cublasStatus_t err = call;                               \
    if (err != CUBLAS_STATUS_SUCCESS)                              \
    {                                                              \
      const char *errStr = "<unknown>";                            \
      switch (err)                                                 \
      {                                                            \
      case CUBLAS_STATUS_SUCCESS:                                  \
        errStr = "CUBLAS_STATUS_SUCCESS";                          \
      case CUBLAS_STATUS_NOT_INITIALIZED:                          \
        errStr = "CUBLAS_STATUS_NOT_INITIALIZED";                  \
      case CUBLAS_STATUS_ALLOC_FAILED:                             \
        errStr = "CUBLAS_STATUS_ALLOC_FAILED";                     \
      case CUBLAS_STATUS_INVALID_VALUE:                            \
        errStr = "CUBLAS_STATUS_INVALID_VALUE";                    \
      case CUBLAS_STATUS_ARCH_MISMATCH:                            \
        errStr = "CUBLAS_STATUS_ARCH_MISMATCH";                    \
      case CUBLAS_STATUS_MAPPING_ERROR:                            \
        errStr = "CUBLAS_STATUS_MAPPING_ERROR";                    \
      case CUBLAS_STATUS_EXECUTION_FAILED:                         \
        errStr = "CUBLAS_STATUS_EXECUTION_FAILED";                 \
      case CUBLAS_STATUS_INTERNAL_ERROR:                           \
        errStr = "CUBLAS_STATUS_INTERNAL_ERROR";                   \
      case CUBLAS_STATUS_NOT_SUPPORTED:                            \
        errStr = "CUBLAS_STATUS_NOT_SUPPORTED";                    \
      }                                                            \
                                                                   \
      printf("%s in %s at line %d\n", errStr, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                          \
    }                                                              \
  }

using coordinate_type = float;

// compute matrix multiplication on CPU
void matrixmult(const coordinate_type *__restrict__ M, const coordinate_type *__restrict__ N, coordinate_type *__restrict__ P, const int numMRows, const int numMColumns, const int numNColumns)
{
  for (int i = 0; i < numMRows; i++)
  {
    for (int j = 0; j < numNColumns; j++)
    {
      P[i * numMColumns + j] = {0};
      for (int k = 0; k < numMColumns; k++)
        P[i * numMColumns + j] += M[i * numMColumns + k] * N[k * numNColumns + j];
    }
  }
}

// compute matrix multiplication on GPU
// Naive approach
__global__ void basic_matrixmult(const coordinate_type *__restrict__ M,
                                 const coordinate_type *__restrict__ N,
                                 coordinate_type *__restrict__ P,
                                 const int numMRows,
                                 const int numMColumns,
                                 const int numNColumns)
{
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < numMRows && j < numNColumns)
  {
    int sum = 0;
    for (int k = 0; k < numMColumns; k++)
      sum += M[i * numMColumns + k] * N[k * numNColumns + j];
    P[i * numMColumns + j] = sum;
  }
}

// compute tiled matrix multiplication on GPU
// Tiled approach
__global__ void tiled_matrixmult(const coordinate_type *__restrict__ M,
                                 const coordinate_type *__restrict__ N,
                                 coordinate_type *__restrict__ P,
                                 const int numMRows,
                                 const int numMColumns,
                                 const int numNColumns)
{
  __shared__ coordinate_type tM[TILE_WIDTH][TILE_WIDTH];
  __shared__ coordinate_type tN[TILE_WIDTH][TILE_WIDTH];

  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int numNRows = numMColumns;
  int sum = 0;

  for (int m = 0; m < (numMColumns - 1) / TILE_WIDTH + 1; ++m)
  {
    tM[threadIdx.y][threadIdx.x] = i < numMRows && m * TILE_WIDTH + threadIdx.x < numMColumns
                                       ? M[i * numMColumns + m * TILE_WIDTH + threadIdx.x]
                                       : 0;
    tN[threadIdx.y][threadIdx.x] = m * TILE_WIDTH + threadIdx.y < numNRows && j < numNColumns
                                       ? N[(m * TILE_WIDTH + threadIdx.y) * numNColumns + j]
                                       : 0;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++)
      sum += tM[threadIdx.y][k] * tN[k][threadIdx.x];
    __syncthreads();
  }

  if (i < numMRows && j < numNColumns)
    P[i * numMColumns + j] = sum;
}

int main(int argc, char **argv)
{
  coordinate_type *h_A;     // The A matrix
  coordinate_type *h_B;     // The B matrix
  coordinate_type *h_C;     // The output C matrix
  coordinate_type *h_C_cpu; // The output C matrix computed on the CPU
  coordinate_type *d_A;
  coordinate_type *d_B;
  coordinate_type *d_C;
  unsigned int numARows;    // number of rows in the matrix A
  unsigned int numAColumns; // number of columns in the matrix A
  unsigned int numBRows;    // number of rows in the matrix B
  unsigned int numBColumns; // number of columns in the matrix B
  unsigned int numCRows;
  unsigned int numCColumns;
  int ok, i;

  if (argc != 5)
  {
    printf("Please specify sizes (#rows and #columns) of matrix A and B\n");
    return 0;
  }
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);

  if (numAColumns != numBRows)
  {
    printf("# colums of A is different from the number of rows of B\n");
    return 0;
  }

  // compute output matrix size
  numCRows = numARows;
  numCColumns = numBColumns;

  // allocate memory for the three matrices
  h_A = (coordinate_type *)malloc(sizeof(coordinate_type) * numARows * numAColumns);
  if (!h_A)
  {
    printf("Error: malloc failed\n");
    return 1;
  }
  h_B = (coordinate_type *)malloc(sizeof(coordinate_type) * numBRows * numBColumns);
  if (!h_B)
  {
    printf("Error: malloc failed\n");
    free(h_A);
    return 1;
  }
  h_C = (coordinate_type *)malloc(sizeof(coordinate_type) * numCRows * numCColumns);
  if (!h_C)
  {
    printf("Error: malloc failed\n");
    free(h_A);
    free(h_B);
    return 1;
  }
  h_C_cpu = (coordinate_type *)malloc(sizeof(coordinate_type) * numCRows * numCColumns);
  if (!h_C_cpu)
  {
    printf("Error: malloc failed\n");
    free(h_A);
    free(h_B);
    free(h_C);
    return 1;
  }
  // initialize input matrices
  srand(0);
  for (i = 0; i < numARows * numAColumns; i++)
    h_A[i] = rand() % MAXVAL;
  for (i = 0; i < numBRows * numBColumns; i++)
    h_B[i] = rand() % MAXVAL;

  // execute on CPU
  matrixmult(h_A, h_B, h_C_cpu, numARows, numAColumns, numBColumns);

  // allocate device memory and transfer data
  CHECK(cudaMalloc(&d_A, numARows * numAColumns * sizeof(coordinate_type)));
  CHECK(cudaMalloc(&d_B, numBRows * numBColumns * sizeof(coordinate_type)));
  CHECK(cudaMalloc(&d_C, numCRows * numCColumns * sizeof(coordinate_type)));
  CHECK(cudaMemcpy(d_A, h_A, numARows * numAColumns * sizeof(coordinate_type), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, numBRows * numBColumns * sizeof(coordinate_type), cudaMemcpyHostToDevice));

  // execute on GPU 1
  dim3 blockDim(BLOCKDIM, BLOCKDIM);
  dim3 gridDim(ceil(((float)numBColumns) / blockDim.x), ceil(((float)numARows) / blockDim.y));

  basic_matrixmult<<<gridDim, blockDim>>>(d_A, d_B, d_C, numARows, numAColumns, numBColumns);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  // transfer results 1
  CHECK(cudaMemcpy(h_C, d_C, numCRows * numCColumns * sizeof(coordinate_type), cudaMemcpyDeviceToHost));

  // check results
  for (i = 0, ok = 1; i < numCRows * numCColumns; i++)
    if (h_C[i] != h_C_cpu[i])
    {
      printf("Different num %f %f at %d\n", h_C[i], h_C_cpu[i], i);
      ok = 0;
    }
  printf("Result: %s\n", ok ? "OK" : "NO");

  // execute 2
  tiled_matrixmult<<<gridDim, blockDim>>>(d_A, d_B, d_C, numARows, numAColumns, numBColumns);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  // transfer results 2
  CHECK(cudaMemcpy(h_C, d_C, numCRows * numCColumns * sizeof(coordinate_type), cudaMemcpyDeviceToHost));

  // check results
  for (i = 0, ok = 1; i < numCRows * numCColumns; i++)
    if (h_C[i] != h_C_cpu[i])
    {
      printf("Different num %f %f at %d\n", h_C[i], h_C_cpu[i], i);
      ok = 0;
    }
  printf("Result: %s\n", ok ? "OK" : "NO");

  // execute 3
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  const coordinate_type alpha{1};
  const coordinate_type beta{0};
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, numBColumns, numARows, numAColumns, &alpha, d_B, numBColumns, d_A, numAColumns, &beta, d_C, numBColumns));
  CHECK(cudaDeviceSynchronize());
  CHECK_CUBLAS(cublasDestroy(handle));

  // transfer results 3
  CHECK(cudaMemcpy(h_C, d_C, numCRows * numCColumns * sizeof(coordinate_type), cudaMemcpyDeviceToHost));

  // check results
  for (i = 0, ok = 1; i < numCRows * numCColumns; i++)
    if (h_C[i] != h_C_cpu[i])
    {
      printf("Different num %f %f at %d\n", h_C[i], h_C_cpu[i], i);
      ok = 0;
    }
  printf("Result: %s\n", ok ? "OK" : "NO");

  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_C));

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_cpu);

  return 0;
}
