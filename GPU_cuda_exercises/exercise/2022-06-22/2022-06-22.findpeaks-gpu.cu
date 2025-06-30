/*
 * The kernel function to accelerate receives in input a vector of positive integers, called A,
 * together with its size, and a second empty vector of integers, B, of the same size.
 * For each element i in A, the function saves in B[i] the value 1 if A[i] is greater than all the
 * neighbor values with an index between (i-DIST) and (i+DIST), bounds included and if they exist;
 * 0 otherwise. DIST is a constant value defined with a macro.
 * The main function is a dummy program that receives as an argument the vector size, instantiates and
 * populates randomly A, invokes the above function, and shows results.
 */

#include <stdio.h>
#include <stdlib.h>

#define MAXVAL 100
#define DIST 10
#define BLOCK_DIM 32
#define TILE_DIM (BLOCK_DIM + 2 * DIST)

void printV(int *V, int num);
void compute(int *V, int *R, int num);

// display a vector of numbers on the screen
void printV(int *V, int num)
{
  int i;
  for (i = 0; i < num; i++)
    printf("%3d(%d) ", V[i], i);
  printf("\n");
}

// kernel function: identify peaks in the vector
void compute(int *V, int *R, int num)
{
  int i, j, ok;
  for (i = 0; i < num; i++)
  {
    for (j = -DIST, ok = 1; j <= DIST; j++)
    {
      if (i + j >= 0 && i + j < num && j != 0 && V[i] <= V[i + j])
        ok = 0;
    }
    R[i] = ok;
  }
}

__global__ void compute_gpu(int *V, int *R, int num)
{
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  int j, ok;
  if (tid < num)
  {
    for (j = -DIST, ok = 1; j <= DIST; j++)
    {
      if (tid + j >= 0 && tid + j < num && j != 0 && V[tid] <= V[tid + j])
        ok = 0;
    }
    R[tid] = ok;
  }
}

__global__ void compute_shared_gpu(int *V, int *R, int num)
{
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ int input_s[TILE_DIM];
  input_s[DIST + threadIdx.x] = V[tid];
  if (threadIdx.x < DIST)
  {
    if (blockIdx.x > 0)
      input_s[threadIdx.x] = V[tid - DIST];
    else
      input_s[threadIdx.x] = 0;
  }
  if (threadIdx.x >= blockDim.x - DIST)
  {
    if (blockIdx.x < gridDim.x - 1)
      input_s[threadIdx.x + DIST * 2] = V[tid + DIST];
    else
      input_s[DIST * 2 + threadIdx.x] = 0;
  }

  __syncthreads();

  int j, ok;
  if (tid < num)
  {
    for (j = threadIdx.x, ok = 1; j <= DIST * 2 + threadIdx.x; j++)
    {
      if (j != threadIdx.x + DIST && input_s[threadIdx.x + DIST] <= input_s[j])
        ok = 0;
    }
    R[tid] = ok;
  }
}

int main(int argc, char **argv)
{
  int *A;
  int *B;
  int dim;
  int i;

  // read arguments
  if (argc != 2)
  {
    printf("Please specify sizes of the input vector\n");
    return 0;
  }
  dim = atoi(argv[1]);

  // allocate memory for the three vectors
  A = (int *)malloc(sizeof(int) * dim);
  if (!A)
  {
    printf("Error: malloc failed\n");
    return 1;
  }
  B = (int *)malloc(sizeof(int) * dim);
  if (!B)
  {
    printf("Error: malloc failed\n");
    return 1;
  }

  // initialize input vectors
  srand(0);
  for (i = 0; i < dim; i++)
    A[i] = rand() % MAXVAL + 1;

  // execute on CPU
  compute(A, B, dim);

  // print results
  printf("CPU res:\n");
  printV(A, dim);
  printV(B, dim);

  int *d_A, *d_B;
  cudaMalloc(&d_A, sizeof(int) * dim);
  cudaMalloc(&d_B, sizeof(int) * dim);
  cudaMemcpy(d_A, A, sizeof(int) * dim, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BLOCK_DIM);
  dim3 blocksPerGrid((dim - 1) / BLOCK_DIM + 1);
  compute_shared_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, dim);
  cudaDeviceSynchronize();

  int *gpu_res = (int *)malloc(sizeof(int) * dim);
  cudaMemcpy(gpu_res, d_B, sizeof(int) * dim, cudaMemcpyDeviceToHost);

  printf("GPU res:\n");
  printV(gpu_res, dim);
  for (int k = 0; k < dim; k++)
  {
    if (gpu_res[k] != B[k])
    {
      printf("CPU and GPU results are NOT correct...\n");
      exit(EXIT_FAILURE);
    }
  }

  printf("ALL OK...\n");
  cudaDeviceReset();

  free(A);
  free(B);

  return 0;
}
