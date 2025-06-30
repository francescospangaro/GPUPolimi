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

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <sys/time.h>
#endif

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
#define BLOCKDIM 32
#define GRIDDIM 80
#define DIST (10 + BLOCKDIM)
#define INT_CEIL(x, y) (((x) + (y) - 1) / (y))

#define TO_STRING_HELPER(X) #X
#define TO_STRING(X) TO_STRING_HELPER(X)

// Need to do it like this, otherwise nvcc does not compile it properly for some reason
#define UNROLL(x) _Pragma(TO_STRING(unroll(x)))

void printV(int *V, int num);
double get_time();

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

// kernel function: identify peaks in the vector
__global__ void computeKernel(int const *const __restrict__ V,
                              int *const __restrict__ R,
                              int const num)
{
  int const i = blockIdx.x * blockDim.x + threadIdx.x;

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

// kernel function: identify peaks in the vector
__global__ void computeKernel2d(int const *const __restrict__ V,
                                int *const __restrict__ R,
                                int const num)
{
  int const i0 = blockIdx.x * blockDim.x + threadIdx.x;
  int const j0 = blockIdx.y * blockDim.y + threadIdx.y;

  for (int i = i0; i < num; i += blockDim.x * gridDim.x)
  {
    if (j0 == 0)
      R[i] = 1;
    __syncthreads();

    int V_i = V[i], ok = 1;
    UNROLL(INT_CEIL(2 * DIST + 1, BLOCKDIM))
    for (int j = j0 - DIST; j <= DIST; j += blockDim.y)
      if (i + j >= 0 && i + j < num && j != 0 && V_i <= V[i + j])
        ok = 0;

    if (!ok)
      R[i] = 0;
  }
}

// kernel function: identify peaks in the vector
__global__ void computeKernelSharedMem(int const *const __restrict__ V,
                                       int *const __restrict__ R,
                                       int const num)
{
  int const i = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int V_tile[BLOCKDIM + 2 * DIST];
  int *V_off = &V_tile[DIST];

  UNROLL(INT_CEIL(BLOCKDIM + 2 * DIST, BLOCKDIM))
  for (int off = threadIdx.x - DIST; off < BLOCKDIM + DIST; off += blockDim.x)
  {
    int const i0 = blockIdx.x * blockDim.x + off;
    if (i0 >= 0 && i0 < num)
      V_off[off] = V[i0];
  }
  /*
  if (i < num)
    V_off[threadIdx.x] = V[i];
  if (threadIdx.x < DIST && i - DIST >= 0)
    V_off[threadIdx.x - DIST] = V[i - DIST];
  if (threadIdx.x >= blockDim.x - DIST && i + DIST < num)
    V_off[threadIdx.x + DIST] = V[i + DIST];
  */
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

// kernel function: identify peaks in the vector
__global__ void computeKernelSharedMem2d(int const *const __restrict__ V,
                                         int *const __restrict__ R,
                                         int const num)
{
  int const i0 = blockIdx.x * blockDim.x + threadIdx.x;
  int const j0 = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ int V_tile[BLOCKDIM + 2 * DIST];
  int *V_off = &V_tile[DIST];

  for (int stride_off = 0; blockIdx.x * blockDim.x + stride_off < num; stride_off += blockDim.x * gridDim.x)
  {
    int const i = i0 + stride_off;
    if (j0 == 0)
      R[i] = 1;

    int const linear_thread_idx = threadIdx.y * blockDim.y + (threadIdx.x + stride_off);
    UNROLL(INT_CEIL(BLOCKDIM + 2 * DIST, BLOCKDIM * BLOCKDIM))
    for (int off = linear_thread_idx - DIST; off < blockDim.x + DIST; off += blockDim.x * blockDim.y)
    {
      int const copy_idx = blockIdx.x * blockDim.x + off;
      if (copy_idx >= 0 && copy_idx < num)
        V_off[off] = V[copy_idx];
    }
    __syncthreads();

    int V_i = V[i], ok = 1;
    UNROLL(INT_CEIL(2 * DIST + 1, BLOCKDIM))
    for (int j = j0 - DIST; j <= DIST; j += blockDim.y)
    {
      if (i + j >= 0 && i + j < num && j != 0 && V_i <= V[i + j])
        ok = 0;
    }

    if (!ok)
      R[i] = ok;
  }
}

bool arrayEquals(int *a1, int *a2, int n)
{
  for (int i = 0; i < n; ++i)
    if (a1[i] != a2[i])
    {
      printf("Wrong: %d, %d at %d\n", a1[i], a2[i], i);
      return false;
    }
  return true;
}

int main(int argc, char **argv)
{
  int *A;
  int *B;
  int dim;
  int i;
  int nIters = 10;

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
  double cpu_start = get_time();
  compute(A, B, dim);
  double cpu_end = get_time();
  printf("CPU in %.3fms\n", (cpu_end - cpu_start) * 1000);

  // print results
  if (dim < 100)
  {
    printV(A, dim);
    printV(B, dim);
  }

  // execute on GPU
  int *A_d, *B1_d, *B1_h, *B2_d, *B2_h;
  float gpuTime = 0, worstGpuTime = 0;
  cudaEvent_t gpu_start, gpu_end;

  B1_h = (int *)malloc(sizeof(int) * dim);
  if (!B)
  {
    printf("Error: malloc failed\n");
    return 1;
  }

  B2_h = (int *)malloc(sizeof(int) * dim);
  if (!B)
  {
    printf("Error: malloc failed\n");
    return 1;
  }

  CHECK(cudaEventCreate(&gpu_start));
  CHECK(cudaEventCreate(&gpu_end));

  CHECK(cudaMalloc(&A_d, sizeof(int) * dim));
  CHECK(cudaMalloc(&B1_d, sizeof(int) * dim));
  CHECK(cudaMalloc(&B2_d, sizeof(int) * dim));
  CHECK(cudaMemcpy(A_d, A, sizeof(int) * dim, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(BLOCKDIM);
  dim3 numOfBlocks(INT_CEIL(dim, BLOCKDIM));

  // Kernel 1
  for (int n = 0; n < nIters; ++n)
  {
    float currGpuTime;
    CHECK(cudaEventRecord(gpu_start));
    computeKernel<<<numOfBlocks, threadsPerBlock>>>(A_d, B1_d, dim);
    CHECK_KERNELCALL();
    CHECK(cudaEventRecord(gpu_end));
    CHECK(cudaEventSynchronize(gpu_end));
    CHECK(cudaEventElapsedTime(&currGpuTime, gpu_start, gpu_end));

    gpuTime += min(worstGpuTime, currGpuTime);
    worstGpuTime = max(currGpuTime, worstGpuTime);
  }
  CHECK(cudaMemcpy(B1_h, B1_d, sizeof(int) * dim, cudaMemcpyDeviceToHost));
  printf("Kernel 1: %s in %.3fms\n", arrayEquals(B, B1_h, dim) ? "OK" : "KO", gpuTime / nIters);

  // Kernel 2
  for (int n = 0; n < nIters; ++n)
  {
    float currGpuTime;
    CHECK(cudaEventRecord(gpu_start));
    computeKernelSharedMem<<<numOfBlocks, threadsPerBlock>>>(A_d, B2_d, dim);
    CHECK_KERNELCALL();
    CHECK(cudaEventRecord(gpu_end));
    CHECK(cudaEventSynchronize(gpu_end));
    CHECK(cudaEventElapsedTime(&currGpuTime, gpu_start, gpu_end));

    gpuTime += min(worstGpuTime, currGpuTime);
    worstGpuTime = max(currGpuTime, worstGpuTime);
  }
  CHECK(cudaMemcpy(B2_h, B2_d, sizeof(int) * dim, cudaMemcpyDeviceToHost));
  printf("Kernel 2: %s in %.3fms\n", arrayEquals(B, B2_h, dim) ? "OK" : "KO", gpuTime / nIters);

  // Kernel 3
  threadsPerBlock = dim3(BLOCKDIM, BLOCKDIM);
  numOfBlocks = dim3(GRIDDIM, GRIDDIM);

  CHECK(cudaMemset(B1_d, 0x1337, sizeof(int) * dim)); // Make sure to scramble it so we don't accidentally init it properly
  CHECK(cudaMemset(B2_d, 0x1337, sizeof(int) * dim)); // Make sure to scramble it so we don't accidentally init it properly

  for (int n = 0; n < nIters; ++n)
  {
    float currGpuTime;
    CHECK(cudaEventRecord(gpu_start));
    computeKernel2d<<<numOfBlocks, threadsPerBlock>>>(A_d, B1_d, dim);
    CHECK_KERNELCALL();
    CHECK(cudaEventRecord(gpu_end));
    CHECK(cudaEventSynchronize(gpu_end));
    CHECK(cudaEventElapsedTime(&currGpuTime, gpu_start, gpu_end));

    gpuTime += min(worstGpuTime, currGpuTime);
    worstGpuTime = max(currGpuTime, worstGpuTime);
  }
  CHECK(cudaMemcpy(B1_h, B1_d, sizeof(int) * dim, cudaMemcpyDeviceToHost));
  printf("Kernel 3: %s in %.2fms\n", arrayEquals(B, B1_h, dim) ? "OK" : "KO", gpuTime / nIters);

  // Kernel 4
  for (int n = 0; n < nIters; ++n)
  {
    float currGpuTime;
    CHECK(cudaEventRecord(gpu_start));
    computeKernelSharedMem2d<<<numOfBlocks, threadsPerBlock>>>(A_d, B2_d, dim);
    CHECK_KERNELCALL();
    CHECK(cudaEventRecord(gpu_end));
    CHECK(cudaEventSynchronize(gpu_end));
    CHECK(cudaEventElapsedTime(&currGpuTime, gpu_start, gpu_end));

    gpuTime += min(worstGpuTime, currGpuTime);
    worstGpuTime = max(currGpuTime, worstGpuTime);
  }
  CHECK(cudaMemcpy(B2_h, B2_d, sizeof(int) * dim, cudaMemcpyDeviceToHost));
  printf("Kernel 4: %s in %.2fms\n", arrayEquals(B, B2_h, dim) ? "OK" : "KO", gpuTime / nIters);

  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B1_d));
  CHECK(cudaFree(B2_d));

  CHECK(cudaEventDestroy(gpu_end));
  CHECK(cudaEventDestroy(gpu_start));

  free(A);
  free(B);

  return 0;
}

// function to get the time of day in seconds
double get_time()
{
#if defined(_WIN32) || defined(_WIN64)
  FILETIME ft;
  GetSystemTimePreciseAsFileTime(&ft);
  return ((UINT64)(ft.dwLowDateTime) | ((UINT64)(ft.dwHighDateTime) << 32uLL)) * 1e-7;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

// display a vector of numbers on the screen
void printV(int *V, int num)
{
  int i;
  for (i = 0; i < num; i++)
    printf("%3d(%d) ", V[i], i);
  printf("\n");
}
