#include <stdio.h>
#include <stdlib.h>

#define MAXVAL 100
#define DIST 10
#define BLOCK_DIM 32

void compute(int *V, int *R, int num);

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

  if (tid < num)
  {
    int ok = 1;
    for (int j = -DIST; j <= DIST; j++)
    {

      if (tid + j >= 0 && tid + j < num && j != 0 && V[tid] <= V[tid + j])
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
  /*code omitted for the sake of space*/

  // execute on CPU
  compute(A, B, dim);

  // print results
  /*code omitted for the sake of space*/
  int *d_A, *d_B;
  cudaMalloc(&d_A, sizeof(int) * dim);
  cudaMalloc(&d_B, sizeof(int) * dim);
  cudaMemcpy(d_A, A, sizeof(int) * dim, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
  dim3 blocksPerGrid((dim - 1) / BLOCK_DIM, 1, 1);

  compute_gpu<<<threadsPerBlock, blocksPerGrid>>>(d_A, d_B, dim);
  cudaDeviceSynchronize();

  int *gpu_results = (int *)malloc(sizeof(int) * dim);
  cudaMemcpy(gpu_results, d_B, sizeof(int) * dim, cudaMemcpyDeviceToHost);

  // Check if gpu_results == B

  cudaDeviceReset();
  free(A);
  free(B);

  return 0;
}