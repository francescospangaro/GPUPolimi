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

#define MAXVAL 100
#define DIST 10
#define BLOCKDIM 32

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

// display a vector of numbers on the screen
bool arrayEquals(int *a1, int *a2, int size)
{
  for (int i = 0; i < size; i++)
    if (a1[i] != a2[i])
      return false;
  return true;
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

__global__ void compute_kernel(int const *const V,
                               int *const R,
                               int num)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < num)
  {
    int ok = 1;
    for (int j = -DIST; j <= DIST; j++)
    {
      if (i + j >= 0 && i + j < num && j != 0 && V[i] <= V[i + j])
        ok = 0;
    }
    R[i] = ok;
  }
}

__global__ void compute_shared_mem_kernel(int const *const V,
                                          int *const R,
                                          int num)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ int V_s[BLOCKDIM + DIST * 2];
  int *V_off = &V_s[DIST];

  if (i < num)
    V_off[threadIdx.x] = V[i];
  if (threadIdx.x < DIST)
    V_off[threadIdx.x - DIST] = i - DIST >= 0 ? V[i - DIST] : 0;
  if (threadIdx.x >= blockDim.x - DIST)
    V_off[threadIdx.x + DIST] = i + DIST < num ? V[i + DIST] : 0;
  __syncthreads();

  if (i < num)
  {
    int ok = 1;
    for (int j = -DIST; j <= DIST; j++)
    {
      if (i + j >= 0 && i + j < num && j != 0 && V_off[threadIdx.x] <= V_off[threadIdx.x + j])
        ok = 0;
    }
    R[i] = ok;
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

  // execute on GPU
  int *A_d, *B1_d, *B1_h, *B2_d, *B2_h;

  B1_h = (int *)malloc(sizeof(int) * dim);
  if (!B1_h)
  {
    printf("Error: malloc failed\n");
    return 1;
  }

  B2_h = (int *)malloc(sizeof(int) * dim);
  if (!B2_h)
  {
    printf("Error: malloc failed\n");
    return 1;
  }

  CHECK(cudaMalloc(&A_d, dim * sizeof(int)));
  CHECK(cudaMalloc(&B1_d, dim * sizeof(int)));
  CHECK(cudaMalloc(&B2_d, dim * sizeof(int)));
  CHECK(cudaMemcpy(A_d, A, dim * sizeof(int), cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(BLOCKDIM);
  dim3 numOfBlocks((dim + BLOCKDIM - 1) / BLOCKDIM);

  compute_kernel<<<numOfBlocks, threadsPerBlock>>>(A_d, B1_d, dim);
  CHECK_KERNELCALL();

  compute_shared_mem_kernel<<<numOfBlocks, threadsPerBlock>>>(A_d, B2_d, dim);
  CHECK_KERNELCALL();

  CHECK(cudaMemcpy(B1_h, B1_d, dim * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(B2_h, B2_d, dim * sizeof(int), cudaMemcpyDeviceToHost));

  // print results
  printV(A, dim);
  printV(B, dim);
  printf("Kernel 1 %s\n", arrayEquals(B, B1_h, dim) ? "OK" : "KO");
  printf("Kernel 2 %s\n", arrayEquals(B, B2_h, dim) ? "OK" : "KO");

  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B1_d));
  CHECK(cudaFree(B2_d));

  free(A);
  free(B);
  free(B1_h);
  free(B2_h);

  return 0;
}
