#include <stdio.h>
#include <cuda_runtime_api.h>

#define BLOCKDIM 32
#define GRIDIM 32
#define GRID_MULTIPLE 1

#define CHECK(call) \
{ \
  const cudaError_t err = call; \
  if (err != cudaSuccess) { \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} \


#define CHECK_KERNELCALL() \
{ \
  const cudaError_t err = cudaGetLastError();\
  if (err != cudaSuccess) {\
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
  }\
}\


// Kernel to execute the saxpy
__global__
void saxpy(const int n, const float a, float * __restrict__ x, float * __restrict__ y)
{
  const int tid = blockIdx.x*blockDim.x + threadIdx.x;
  const int stride = blockDim.x*gridDim.x;
  for(int index=tid; index<n; index+=stride)
    y[index] = a*x[index] + y[index];
}

int main(void)
{
  const int N = 1<<28;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  CHECK(cudaMalloc(&d_x, N*sizeof(float))); 
  CHECK(cudaMalloc(&d_y, N*sizeof(float)));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  CHECK(cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice));

  // Insert code to play with GRID's dimensions
  saxpy<<<GRIDIM, BLOCKDIM>>>(N, 2.0f, d_x, d_y);
  CHECK_KERNELCALL();

  CHECK(cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  CHECK(cudaFree(d_x));
  CHECK(cudaFree(d_y));
  free(x);
  free(y);
}
