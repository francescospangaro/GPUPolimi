/*
 * Vector addition.
 * Version 0: the sum is performed by a function on the CPU
 */

#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK(call)                                            \
  {                                                            \
    const cudaError_t err = (call);                            \
    if (err != cudaSuccess)                                    \
    {                                                          \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), \
             __FILE__, __LINE__);                              \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

#define CHECK_KERNELCALL()                                     \
  {                                                            \
    const cudaError_t err = cudaGetLastError();                \
    if (err != cudaSuccess)                                    \
    {                                                          \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), \
             __FILE__, __LINE__);                              \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

#define N (1 << 16)
#define BLOCK_SIZE 64

inline double milliseconds()
{
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((float)tp.tv_sec * 1.e+3 + (float)tp.tv_usec * 1.e-3);
}

bool arrayEquals(int *a, int *b, int dim)
{
  for (int i = 0; i < dim; ++i)
  {
    if (a[i] != b[i])
    {
      printf("%d %d %d\n", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

/* CPU function performing vector addition c = a + b */
void vsum(int *a, int *b, int *c, int dim)
{
  int i;
  for (i = 0; i < dim; i++)
    c[i] = a[i] + b[i];
}

/* GPU kernel performing vector addition c = a + b */
__global__ void gpu_vsum(int *a, int *b, int *c, int dim)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim)
    c[i] = a[i] + b[i];
}

int main()
{
  int h_va[N], h_vb[N], h_vc_cpu[N], h_vc_gpu[N];

  /* initialize vectors */
  for (int i = 0; i < N; ++i)
  {
    h_va[i] = i;
    h_vb[i] = N - i;
  }

  /* call CPU function */
  double start = milliseconds();
  vsum(h_va, h_vb, h_vc_cpu, N);
  double elapsed_cpu_millis = milliseconds() - start;

  /* allocate GPU vectors */
  int *d_va, *d_vb, *d_vc;
  CHECK(cudaMalloc(&d_va, sizeof(h_va)));
  CHECK(cudaMalloc(&d_vb, sizeof(h_vb)));
  CHECK(cudaMalloc(&d_vc, sizeof(h_vc_gpu)));

  /* create maarker events */
  cudaEvent_t startMark, stopMark;
  CHECK(cudaEventCreate(&startMark));
  CHECK(cudaEventCreate(&stopMark));

  /* Move memory from host to device */
  CHECK(cudaMemcpy(d_va, h_va, sizeof(h_va), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vb, h_vb, sizeof(h_vb), cudaMemcpyHostToDevice));

  /* launch kernel */
  CHECK(cudaEventRecord(startMark));
  gpu_vsum<<<ceilf(N / (float)BLOCK_SIZE), BLOCK_SIZE>>>(d_va, d_vb, d_vc, N);
  CHECK_KERNELCALL();
  CHECK(cudaEventRecord(stopMark));

  /* copy back result */
  CHECK(cudaMemcpy(h_vc_gpu, d_vc, sizeof(h_vc_gpu), cudaMemcpyDeviceToHost));

  /* calculate elapsed time right after we get synchronized by memcpy */
  float elapsed_gpu_millis;
  CHECK(cudaEventElapsedTime(&elapsed_gpu_millis, startMark, stopMark));

  /* we don't print the results... */
  printf("Done! CPU took: %fms, GPU took: %fms\n", elapsed_cpu_millis, elapsed_gpu_millis);
  printf(arrayEquals(h_vc_cpu, h_vc_gpu, N) ? "Same\n" : "Not the same\n");

  /* destroy events */
  cudaEventDestroy(startMark);
  cudaEventDestroy(stopMark);

  /* free GPU memory */
  cudaFree(d_va);
  cudaFree(d_vb);
  cudaFree(d_vc);

  return 0;
}
