/*
 * The following program elaborates a 2D matrix of integers where each row
 * contains a variable number of elements. In particular, the kernel function
 * receives in input the matrix and sum to each element a value obtained by
 * multiplying the row number by the number of elements in the row. For
 * instance, if we consider the following matrix with 4 rows: 1 2 3 4 1 2 1 1 1
 * 1 1
 * 2
 * The kernel function will compute a new matrix as follows:
 * 1 2 3 4
 * 3 4
 * 11 11 11 11 11
 * 5
 *
 * The described matrix is represented in the program as follows:
 * - An integer array A contains all the elements of the matrix in a linearized
 * way
 * - An integer variable NUMOFELEMS containing the overall number of elements in
 * the matrix
 * - An integer variable ROWS contains the number of rows in the matrix
 * - An integer array COLOFFSETS contains the indexes where the elements of each
 * row starts in A
 * - An integer array COLS contains the length of each row of the matrix
 *
 * Thus, the matrix above is modeled in the program as:
 * A = [1 2 3 4 1 2 1 1 1 1 1 2]
 * NUMOFELEMS = 12
 * ROWS = 4
 * COLOFFSETS = [0, 4, 6, 11]
 * COLS = [4, 2, 5, 1]
 */

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// macros used to generate random data
#define MIN_COLS 5
#define MAX_COLS 10
#define MIN_ROWS 3
#define MAX_ROWS 5
#define MAX_VAL 10

#define BLOCKSIZEX 32
#define BLOCKSIZEY 32

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                                  \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,       \
             __LINE__);                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_KERNELCALL()                                                     \
  {                                                                            \
    const cudaError_t err = cudaGetLastError();                                \
    if (err != cudaSuccess) {                                                  \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__,       \
             __LINE__);                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

void elaborateMatrix(int *a, int rows, int *colOffsets, int *cols, int *b);
void printM(int *a, int rows, int *colOffsets, int *cols);

__global__ void elaborate_matrix_gpu(int *a, int rows, int *colOffsets,
                                     int *cols, int *b);
__global__ void elaborate_matrix_gpu1(int *a, int rows, int *colOffsets,
                                      int *cols, int *b);
__global__ void elaborate_matrix_gpu2(int *a, int rows, int *colOffsets,
                                      int *cols, int *b);
__global__ void elaborate_matrix_gpu3(int *a, int col, int *b, int coeff);

__global__ void elaborate_matrix_gpu(int *a, int rows, int *colOffsets,
                                     int *cols, int *b) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i < rows && j < cols[i])
    *(b + colOffsets[i] + j) = *(a + colOffsets[i] + j) + i * cols[i];
}

__global__ void elaborate_matrix_gpu1(int *a, int rows, int *colOffsets,
                                      int *cols, int *b) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j, coeff;
  if (i < rows) {
    coeff = i * cols[i];
    for (j = 0; j < cols[i]; j++)
      *(b + colOffsets[i] + j) = *(a + colOffsets[i] + j) + coeff;
  }
}

__global__ void elaborate_matrix_gpu2(int *a, int rows, int *colOffsets,
                                      int *cols, int *b) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int coeff;
  if (i < rows) {
    coeff = i * cols[i];
    elaborate_matrix_gpu3<<<((cols[i] - 1) / BLOCKSIZEY + 1), BLOCKSIZEY>>>(
        a + colOffsets[i], cols[i], b + colOffsets[i], coeff);
  }
}

__global__ void elaborate_matrix_gpu3(int *a, int col, int *b, int coeff) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < col) {
    *(b + 1) = *(a + 1) + coeff;
  }
}

// kernel functions: sum to each element of the matrix a coefficient obtained
// by multiplying the row number by the row index
void elaborateMatrix(int *a, int rows, int *colOffsets, int *cols, int *b) {
  int i, j, coeff;
  for (i = 0; i < rows; i++) {
    coeff = i * cols[i];
    for (j = 0; j < cols[i]; j++)
      *(b + colOffsets[i] + j) = *(a + colOffsets[i] + j) + coeff;
  }
}

// display the matrix on the screen
void printM(int *a, int rows, int *colOffsets, int *cols) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols[i]; j++)
      printf("%3d ", *(a + colOffsets[i] + j));
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  int *a, *b;      // input and output matrices
  int rows;        // number of rows
  int *cols;       // number of columns per each row
  int *colOffsets; // offsets in the data array pointing where each matrix row
                   // starts
  int numOfElems;  // overall number of elements
  int i, j;

  // generate the matrix
  srand(0);
  rows = rand() % (MAX_ROWS - MIN_ROWS) + MIN_ROWS;
  cols = (int *)malloc(sizeof(int) * rows);
  if (!cols) {
    printf("Error on malloc\n");
    return -1;
  }
  for (i = 0; i < rows; i++)
    cols[i] = rand() % (MAX_COLS - MIN_COLS) + MIN_COLS;

  colOffsets = (int *)malloc(sizeof(int) * rows);
  if (!colOffsets) {
    printf("Error on malloc\n");
    return -1;
  }
  for (i = 0, numOfElems = 0; i < rows; i++) {
    colOffsets[i] = numOfElems;
    numOfElems += cols[i];
  }

  a = (int *)malloc(sizeof(int) * numOfElems);
  if (!a) {
    printf("Error on malloc\n");
    return -1;
  }
  for (i = 0; i < numOfElems; i++)
    a[i] = rand() % MAX_VAL;

  b = (int *)malloc(sizeof(int) * numOfElems);
  if (!b) {
    printf("Error on malloc\n");
    return -1;
  }

  // call the kernel function
  elaborateMatrix(a, rows, colOffsets, cols, b);

  // print results
  printM(a, rows, colOffsets, cols);
  printf("\n");
  printM(b, rows, colOffsets, cols);

  // Needed for GPU
  int *a_d, *b_d, *colOffsets_d, *b_GPU, *cols_d, maxcols;
  CHECK(cudaMalloc((void **)&a_d, sizeof(int) * numOfElems));
  CHECK(cudaMalloc((void **)&b_d, sizeof(int) * numOfElems));
  CHECK(cudaMalloc((void **)&colOffsets_d, sizeof(int) * rows));
  CHECK(cudaMalloc((void **)&cols_d, sizeof(int) * rows));

  CHECK(cudaMemcpy(a_d, a, sizeof(int) * numOfElems, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(cols_d, cols, sizeof(int) * rows, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(colOffsets_d, colOffsets, sizeof(int) * numOfElems,
                   cudaMemcpyHostToDevice));

  b_GPU = (int *)malloc(sizeof(int) * numOfElems);
  if (!b_GPU) {
    printf("Error on malloc\n");
    return -1;
  }

  for (i = 1, maxcols = cols[0]; i < rows; i++)
    if (maxcols < cols[i])
      maxcols = cols[i];

  dim3 blocksPerGrid((maxcols - 1) / BLOCKSIZEX + 1,
                     (rows - 1) / BLOCKSIZEY + 1, 1);
  dim3 threadsPerBlock(BLOCKSIZEX, BLOCKSIZEY, 1);

  elaborate_matrix_gpu<<<blocksPerGrid, threadsPerBlock>>>(
      a_d, rows, colOffsets_d, cols_d, b_d);
  CHECK_KERNELCALL();
  cudaDeviceSynchronize();
  CHECK(
      cudaMemcpy(b_GPU, b_d, sizeof(int) * numOfElems, cudaMemcpyDeviceToHost));

  dim3 blocksPerGrid1((rows - 1) / BLOCKSIZEX + 1, 1, 1);
  dim3 threadsPerBlock1(BLOCKSIZEX, 1, 1);

  elaborate_matrix_gpu1<<<blocksPerGrid1, threadsPerBlock1>>>(
      a_d, rows, colOffsets_d, cols_d, b_d);
  CHECK_KERNELCALL();
  cudaDeviceSynchronize();
  CHECK(
      cudaMemcpy(b_GPU, b_d, sizeof(int) * numOfElems, cudaMemcpyDeviceToHost));

  dim3 blocksPerGrid2((rows - 1) / BLOCKSIZEX + 1, 1, 1);
  dim3 threadsPerBlock2(BLOCKSIZEX, 1, 1);

  elaborate_matrix_gpu2<<<blocksPerGrid1, threadsPerBlock1>>>(
      a_d, rows, colOffsets_d, cols_d, b_d);
  CHECK_KERNELCALL();
  cudaDeviceSynchronize();
  CHECK(
      cudaMemcpy(b_GPU, b_d, sizeof(int) * numOfElems, cudaMemcpyDeviceToHost));

  // release memory
  free(a);
  free(b);
  free(cols);
  free(colOffsets);

  CHECK(cudaFree(a_d));
  CHECK(cudaFree(b_d));
  CHECK(cudaFree(colOffsets_d));
  CHECK(cudaFree(cols_d));

  return 0;
}