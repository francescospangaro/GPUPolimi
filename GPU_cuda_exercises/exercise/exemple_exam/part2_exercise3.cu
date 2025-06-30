#include <stdio.h>
#include <stdlib.h>

#define MAXVAL 100
#define DIST 10
#define BLOCK_DIM 32
#define TILE_DIM (BLOCK_DIM + 2 * DIST)

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
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ int tile_s[TILE_DIM];
  tile_s[DIST + threadIdx.x] = V[blockIdx.x * blockDim.x + threadIdx.x];
  if (threadIdx.x < DIST)
  {
    if (blockIdx.x > 0)
      tile_s[threadIdx.x] = V[blockIdx.x * blockDim.x + threadIdx.x - DIST];
    else
      tile_s[threadIdx.x] = 0;
  }
  if (threadIdx.x >= blockDim.x - DIST)
  {
    if (blockIdx.x < gridDim.x - 1)
      tile_s[threadIdx.x + DIST * 2] = V[(blockIdx.x) * blockDim.x + threadIdx.x + DIST];
    else
      tile_s[DIST * 2 + threadIdx.x] = 0;
  }

  __syncthreads();

  if (tid < num)
  {
    for (int j = threadIdx.x, ok = 1; j <= threadIdx.x + DIST * 2; j++)
    {
      if(tid==242) {
        printf("");
      }
      if (j != threadIdx.x + DIST && tile_s[threadIdx.x + DIST] <= tile_s[j])
      {
        ok = 0;
      }
      R[tid] = ok;
    }
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
  B = (int *)malloc(sizeof(int) * dim);

  // initialize input vectors
  srand(0);
  for (i = 0; i < dim; i++)
    A[i] = rand() % MAXVAL + 1;
  /*code omitted for the sake of space*/
  printV(A, dim);
  // execute on CPU
  compute(A, B, dim);

  // print results
  /*code omitted for the sake of space*/
  int *d_A, *d_B;
  cudaMalloc(&d_A, sizeof(int) * dim);
  cudaMalloc(&d_B, sizeof(int) * dim);
  cudaMemcpy(d_A, A, sizeof(int) * dim, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
  dim3 blocksPerGrid((dim - 1) / BLOCK_DIM +1, 1, 1);

  compute_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, dim);
  cudaDeviceSynchronize();

  int *gpu_results = (int *)malloc(sizeof(int) * dim);
  cudaMemcpy(gpu_results, d_B, sizeof(int) * dim, cudaMemcpyDeviceToHost);

  // Check if gpu_results == B
  for (int i = 0; i < dim; i++)
  {
    if (gpu_results[i] != B[i])
    {
      printf("Error in B[%d]:  %d != %d \n", i, B[i], gpu_results[i]);
      exit(EXIT_FAILURE);
    }
  }

  printf("All results correct!\n");

  cudaDeviceReset();
  free(A);
  free(B);

  return 0;
}