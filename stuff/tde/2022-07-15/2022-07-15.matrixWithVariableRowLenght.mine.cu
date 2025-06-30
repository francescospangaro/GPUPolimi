/* The following program elaborates a 2D matrix of integers where each row contains a
 * variable number of elements. In particular, the kernel function receives in input
 * the matrix and sum to each element a value obtained by multiplying the row number
 * by the number of elements in the row.
 * For instance, if we consider the following matrix with 4 rows:
 * 1 2 3 4
 * 1 2
 * 1 1 1 1 1
 * 2
 * The kernel function will compute a new matrix as follows:
 * 1 2 3 4
 * 3 4
 * 11 11 11 11 11
 * 5
 *
 * The described matrix is represented in the program as follows:
 * - An integer array A contains all the elements of the matrix in a linearized way
 * - An integer variable NUMOFELEMS containing the overall number of elements in the matrix
 * - An integer variable ROWS contains the number of rows in the matrix
 * - An integer array COLOFFSETS contains the indexes where the elements of each row starts in A
 * - An integer array COLS contains the length of each row of the matrix
 *
 * Thus, the matrix above is modeled in the program as:
 * A = [1 2 3 4 1 2 1 1 1 1 1 2]
 * NUMOFELEMS = 12
 * ROWS = 4
 * COLOFFSETS = [0, 4, 6, 11]
 * COLS = [4, 2, 5, 1]
 *
 * Compile with
 * nvcc .\2022-07-15.matrixWithVariableRowLenght.mine.cu -lcudadevrt -rdc=true -o .\2022-07-15.matrixWithVariableRowLenght.mine.exe
 */

#include <stdio.h>
#include <stdlib.h>

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

#define BLOCKDIM 32

// macros used to generate random data
#define MIN_COLS 5
#define MAX_COLS 10
#define MIN_ROWS 3
#define MAX_ROWS 5
#define MAX_VAL 10

void elaborateMatrix(int *a, int rows, int *colOffsets, int *cols, int *b);
void printM(int *a, int rows, int *colOffsets, int *cols);

// kernel functions: sum to each element of the matrix a coefficient obtained
// by multiplying the row number by the row index
void elaborateMatrix(int *a, int rows, int *colOffsets, int *cols, int *b)
{
  int i, j, coeff;
  for (i = 0; i < rows; i++)
  {
    coeff = i * cols[i];
    for (j = 0; j < cols[i]; j++)
      *(b + colOffsets[i] + j) = *(a + colOffsets[i] + j) + coeff;
  }
}

__global__ void elaborateMatrixKernel(int const *const __restrict__ a,
                                      int const rows,
                                      int const *const __restrict__ colOffsets,
                                      int const *const __restrict__ cols,
                                      int *const __restrict__ b)
{
  int const i = blockIdx.y * blockDim.y + threadIdx.y;
  int const j = blockIdx.x * blockDim.x + threadIdx.x;

  int numOfCols;
  if (i < rows && j < (numOfCols = cols[i]))
  {
    int coeff = i * numOfCols;
    int colOffset = colOffsets[i];
    b[colOffset + j] = a[colOffset + j] + coeff;
  }
}

__global__ void elaborateMatrixKernelSub(int const *const __restrict__ a,
                                         int const row,
                                         int const colOffset,
                                         int const cols,
                                         int const coeff,
                                         int *const __restrict__ b)
{
  int const j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < cols)
    b[colOffset + j] = a[colOffset + j] + coeff;
}

__global__ void elaborateMatrixKernel2(int const *const __restrict__ a,
                                       int const rows,
                                       int const *const __restrict__ colOffsets,
                                       int const *const __restrict__ cols,
                                       int *const __restrict__ b)
{
  int const i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < rows)
  {
    int numOfCols = cols[i];
    int colOffset = colOffsets[i];
    elaborateMatrixKernelSub<<<(numOfCols + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(a, rows, colOffset, numOfCols, i * numOfCols, b);
  }
}

// display the matrix on the screen
void printM(int *a, int rows, int *colOffsets, int *cols)
{
  int i, j;
  for (i = 0; i < rows; i++)
  {
    for (j = 0; j < cols[i]; j++)
      printf("%3d ", *(a + colOffsets[i] + j));
    printf("\n");
  }
}

bool arrayEquals(int *a1, int *a2, int size)
{
  for (int i = 0; i < size; ++i)
    if (a1[i] != a2[i])
      return false;
  return true;
}

int main(int argc, char *argv[])
{
  int *a, *b;      // input and output matrices
  int rows;        // number of rows
  int *cols;       // number of columns per each row
  int *colOffsets; // offsets in the data array pointing where each matrix row starts
  int maxCols;     // maximum number of cols of any row in this matrix
  int numOfElems;  // overall number of elements
  int i;

  // generate the matrix
  srand(0);
  rows = rand() % (MAX_ROWS - MIN_ROWS) + MIN_ROWS;
  cols = (int *)malloc(sizeof(int) * rows);
  if (!cols)
  {
    printf("Error on malloc\n");
    return -1;
  }
  for (i = 0; i < rows; i++)
    cols[i] = rand() % (MAX_COLS - MIN_COLS) + MIN_COLS;

  colOffsets = (int *)malloc(sizeof(int) * rows);
  if (!colOffsets)
  {
    printf("Error on malloc\n");
    return -1;
  }
  for (i = 0, numOfElems = 0; i < rows; i++)
  {
    colOffsets[i] = numOfElems;
    numOfElems += cols[i];
  }

  a = (int *)malloc(sizeof(int) * numOfElems);
  if (!a)
  {
    printf("Error on malloc\n");
    return -1;
  }
  for (i = 0; i < numOfElems; i++)
    a[i] = rand() % MAX_VAL;

  b = (int *)malloc(sizeof(int) * numOfElems);
  if (!b)
  {
    printf("Error on malloc\n");
    return -1;
  }

  maxCols = 0;
  for (i = 1; i < rows; ++i)
  {
    if (maxCols < cols[i])
      maxCols = cols[i];
  }

  // call the kernel function
  elaborateMatrix(a, rows, colOffsets, cols, b);

  // call the kernel on GPU
  int *a_d, *b1_d, *b2_d, *b1_h, *b2_h; // input and output matrices
  int *cols_d;                          // number of columns per each row
  int *colOffsets_d;                    // offsets in the data array pointing where each matrix row starts

  b1_h = (int *)malloc(sizeof(int) * numOfElems);
  if (!b1_h)
  {
    printf("Error on malloc\n");
    return -1;
  }

  b2_h = (int *)malloc(sizeof(int) * numOfElems);
  if (!b2_h)
  {
    printf("Error on malloc\n");
    return -1;
  }

  CHECK(cudaMalloc(&a_d, sizeof(int) * numOfElems));
  CHECK(cudaMalloc(&b1_d, sizeof(int) * numOfElems));
  CHECK(cudaMalloc(&b2_d, sizeof(int) * numOfElems));
  CHECK(cudaMalloc(&cols_d, sizeof(int) * rows));
  CHECK(cudaMalloc(&colOffsets_d, sizeof(int) * rows));

  CHECK(cudaMemcpy(a_d, a, sizeof(int) * numOfElems, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(cols_d, cols, sizeof(int) * rows, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(colOffsets_d, colOffsets, sizeof(int) * rows, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock1(BLOCKDIM, BLOCKDIM);
  dim3 numOfBlocks1((maxCols + BLOCKDIM - 1) / BLOCKDIM, (rows + BLOCKDIM - 1) / BLOCKDIM);
  elaborateMatrixKernel<<<numOfBlocks1, threadsPerBlock1>>>(a_d, rows, colOffsets_d, cols_d, b1_d);
  CHECK_KERNELCALL();

  dim3 threadsPerBlock2(BLOCKDIM);
  dim3 numOfBlocks2((rows + BLOCKDIM - 1) / BLOCKDIM);
  elaborateMatrixKernel2<<<numOfBlocks2, threadsPerBlock2>>>(a_d, rows, colOffsets_d, cols_d, b2_d);
  CHECK_KERNELCALL();

  CHECK(cudaMemcpy(b1_h, b1_d, sizeof(int) * numOfElems, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(b2_h, b2_d, sizeof(int) * numOfElems, cudaMemcpyDeviceToHost));

  CHECK(cudaFree(a_d));
  CHECK(cudaFree(b1_d));
  CHECK(cudaFree(b2_d));
  CHECK(cudaFree(cols_d));
  CHECK(cudaFree(colOffsets_d));

  // print results
  printM(a, rows, colOffsets, cols);
  printf("\n");
  printM(b, rows, colOffsets, cols);
  printf("\n");
  printM(b1_h, rows, colOffsets, cols);
  printf("\n");
  printM(b2_h, rows, colOffsets, cols);
  printf("\n");

  printf("Kernel 1: %s\n", arrayEquals(b, b1_h, numOfElems) ? "OK" : "KO");
  printf("Kernel 2: %s\n", arrayEquals(b, b2_h, numOfElems) ? "OK" : "KO");

  // release memory
  free(a);
  free(b);
  free(b1_h);
  free(b2_h);
  free(cols);
  free(colOffsets);

  return 0;
}