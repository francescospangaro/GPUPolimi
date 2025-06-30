/* In physics, the n-body simulation is the simulation of the motion of a set of M
different objects due to the gravitational interaction among them. The following
program implement an N-body simulation. Each object to be simulated is modeled
by means of the Body_t struct that contains fields for the position and the speed
in a 3D space; data of N different objects is stored in an Body_t array.
The simulation starts by computing randomly  the initial position and speed of each
object. Then a number of time steps are simulated; at each time step the bodyForce()
function computes the new speed of each object due to the gravitational interaction
and updateBodyPositions() functions computes the new position of each object.
*/

#include <math.h>
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

#define BLOCKDIM 32

#ifdef DEBUG
#define SOFTENING 1
#define SQRTF(x) (x)
#else
#define SOFTENING 1e-9f
#define SQRTF(x) sqrtf(x)
#endif

typedef struct
{
  float x, y, z, vx, vy, vz;
} Body_t;

double get_time();
bool bodiesSimilar(Body_t *b1, Body_t *b2, int n);

void randomizeBodies(float *data, int n);
void bodyForce(Body_t *p, float dt, int n);
void updateBodyPositions(Body_t *p, float dt, int n);

void randomizeBodies(float *data, int n)
{
  for (int i = 0; i < n; i++)
  {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body_t *p, float dt, int n)
{
  for (int i = 0; i < n; i++)
  {
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int j = 0; j < n; j++)
    {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      float invDist = 1.0f / SQRTF(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
  }
}

void updateBodyPositions(Body_t *p, float dt, int n)
{
  for (int i = 0; i < n; i++)
  {
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
  }
}

__global__ void bodyForceKernel1(Body_t *const __restrict__ p,
                                 float const dt,
                                 int const n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    Body_t currBody = p[i];
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int j = 0; j < n; j++)
    {
      Body_t otherBody = p[j];
      float dx = otherBody.x - currBody.x;
      float dy = otherBody.y - currBody.y;
      float dz = otherBody.z - currBody.z;
      float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      float invDist = 1.0f / SQRTF(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
  }
}

__global__ void bodyForceKernel2(Body_t *const __restrict__ p,
                                 float const dt,
                                 int const n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ Body_t p_tile[BLOCKDIM];

  Body_t currBody;
  if (i < n)
    currBody = p[i];

  float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

  for (int tile = 0; tile < (n + BLOCKDIM - 1) / BLOCKDIM; tile++)
  {
    if (tile * BLOCKDIM + threadIdx.x < n)
      p_tile[threadIdx.x] = p[tile * BLOCKDIM + threadIdx.x];
    __syncthreads();

    if (i < n)
    {
      for (int j = 0; j < BLOCKDIM && tile * BLOCKDIM + j < n; ++j)
      {
        Body_t otherBody = p_tile[j];
        float dx = otherBody.x - currBody.x;
        float dy = otherBody.y - currBody.y;
        float dz = otherBody.z - currBody.z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = 1.0f / SQRTF(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
      }
    }
    __syncthreads();
  }

  if (i < n)
  {
    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
  }
}

__global__ void updateBodyPositionsKernel(Body_t *const __restrict__ p,
                                          float const dt,
                                          int const n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
  }
}

int main(const int argc, const char **argv)
{
  int nBodies = 30000;
  if (argc > 1)
    nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  double cpu_start, cpu_end;

  Body_t *p_orig = (Body_t *)malloc(nBodies * sizeof(Body_t));
  if (!p_orig)
  {
    printf("Malloc failed\n");
    return 1;
  }

  Body_t *p = (Body_t *)malloc(nBodies * sizeof(Body_t));
  if (!p)
  {
    printf("Malloc failed\n");
    return 1;
  }

  randomizeBodies((float *)p_orig, (sizeof(Body_t) / sizeof(float)) * nBodies); // Init position / speed data
  memcpy(p, p_orig, nBodies * sizeof(Body_t));

  cpu_start = get_time();
  for (int iter = 1; iter <= nIters; iter++)
  {
    bodyForce(p, dt, nBodies);           // compute interbody forces
    updateBodyPositions(p, dt, nBodies); // integrate position
  }
  cpu_end = get_time();
  double avgTime = (cpu_end - cpu_start) / nIters;

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

  // GPU

  Body_t *p_d, *p_h;
  p_h = (Body_t *)malloc(nBodies * sizeof(Body_t));
  if (!p_h)
  {
    printf("Malloc failed\n");
    return 1;
  }

  CHECK(cudaMalloc(&p_d, nBodies * sizeof(Body_t)));

  float gpuTime;
  cudaEvent_t gpu_start, gpu_end;

  CHECK(cudaEventCreate(&gpu_start));
  CHECK(cudaEventCreate(&gpu_end));

  dim3 threadsPerBlock(BLOCKDIM);
  dim3 numOfBlocks((nBodies + BLOCKDIM - 1) / BLOCKDIM);

  // Execute on GPU Kernel 1

  CHECK(cudaMemcpy(p_d, p_orig, nBodies * sizeof(Body_t), cudaMemcpyHostToDevice));
  CHECK(cudaEventRecord(gpu_start));
  for (int iter = 1; iter <= nIters; iter++)
  {
    bodyForceKernel1<<<numOfBlocks, threadsPerBlock>>>(p_d, dt, nBodies); // compute interbody forces
    CHECK_KERNELCALL();
    updateBodyPositionsKernel<<<numOfBlocks, threadsPerBlock>>>(p_d, dt, nBodies); // integrate position
    CHECK_KERNELCALL();
  }
  CHECK(cudaEventRecord(gpu_end));
  CHECK(cudaEventSynchronize(gpu_end));
  CHECK(cudaEventElapsedTime(&gpuTime, gpu_start, gpu_end));

  CHECK(cudaMemcpy(p_h, p_d, nBodies * sizeof(Body_t), cudaMemcpyDeviceToHost));

  double gpuAvgTime = gpuTime / 1000 / nIters;
  printf("GPU 1 %s, %d Bodies: average %0.3f Billion Interactions / second\n",
         bodiesSimilar(p, p_h, nBodies) ? "OK" : "KO",
         nBodies, 1e-9 * nBodies * nBodies / gpuAvgTime);
  // cudmemcpy(p, p_h, nBodies * sizeof(Body_t));

  // Execute on GPU Kernel 2

  CHECK(cudaMemcpy(p_d, p_orig, nBodies * sizeof(Body_t), cudaMemcpyHostToDevice));
  CHECK(cudaEventRecord(gpu_start));
  for (int iter = 1; iter <= nIters; iter++)
  {
    bodyForceKernel2<<<numOfBlocks, threadsPerBlock>>>(p_d, dt, nBodies); // compute interbody forces
    CHECK_KERNELCALL();
    updateBodyPositionsKernel<<<numOfBlocks, threadsPerBlock>>>(p_d, dt, nBodies); // integrate position
    CHECK_KERNELCALL();
  }
  CHECK(cudaEventRecord(gpu_end));
  CHECK(cudaEventSynchronize(gpu_end));
  CHECK(cudaEventElapsedTime(&gpuTime, gpu_start, gpu_end));

  CHECK(cudaMemcpy(p_h, p_d, nBodies * sizeof(Body_t), cudaMemcpyDeviceToHost));

  gpuAvgTime = gpuTime / 1000 / nIters;
  printf("GPU 2 %s, %d Bodies: average %0.3f Billion Interactions / second\n",
         bodiesSimilar(p, p_h, nBodies) ? "OK" : "KO",
         nBodies, 1e-9 * nBodies * nBodies / gpuAvgTime);

  // Free stuff

  CHECK(cudaEventDestroy(gpu_start));
  CHECK(cudaEventDestroy(gpu_end));

  CHECK(cudaFree(p_d));

  free(p);
  free(p_h);

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

bool bodiesSimilar(Body_t *b1, Body_t *b2, int n)
{
  // From the official Nvidia course Python test runner, which used this problem as a final assessment:
  // "Floating point calculations between the CPU and GPU vary, so in order to
  // be able to assess both CPU - only and GPU code we compare the floating point
  // arrays within a tolerance of less than 1 % of values differ by 1 or more."

  int totalFloats = (n * (sizeof(Body_t) / sizeof(float)));
  int wrongFloats = 0;
  for (int i = 0; i < n; ++i)
  {
    if (abs(b1[i].x - b2[i].x) >= 1.0f)
      wrongFloats++;
    if (abs(b1[i].y - b2[i].y) >= 1.0f)
      wrongFloats++;
    if (abs(b1[i].z - b2[i].z) >= 1.0f)
      wrongFloats++;
    if (abs(b1[i].vx - b2[i].vx) >= 1.0f)
      wrongFloats++;
    if (abs(b1[i].vy - b2[i].vy) >= 1.0f)
      wrongFloats++;
    if (abs(b1[i].vz - b2[i].vz) >= 1.0f)
      wrongFloats++;
  }

  return (double)wrongFloats / totalFloats < 0.1;
}
