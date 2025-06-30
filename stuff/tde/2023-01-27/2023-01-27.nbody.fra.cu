/* In physics, the n-body simulation is the simulation of the motion of a set of
M different objects due to the gravitational interaction among them. The
following program implement an N-body simulation. Each object to be simulated is
modeled by means of the Body_t struct that contains fields for the position and
the speed in a 3D space; data of N different objects is stored in an Body_t
array. The simulation starts by computing randomly  the initial position and
speed of each object. Then a number of time steps are simulated; at each time
step the bodyForce() function computes the new speed of each object due to the
gravitational interaction and updateBodyPositions() functions computes the new
position of each object.
*/

#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCKSIZE 32

#define SOFTENING 1e-9f

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

typedef struct {
  float x, y, z, vx, vy, vz;
} Body_t;

double get_time();
void randomizeBodies(float *data, int n);
void bodyForce(Body_t *p, float dt, int n);
void updateBodyPositions(Body_t *p, float dt, int n);
__global__ void bodyForceKernel(Body_t *p, float dt, int n);
__global__ void updateBodyPositionsKernel(Body_t *p, float dt, int n);

__global__ void bodyForceKernel(Body_t *p, float dt, int n) {
  // We can't parallelize both i and j, since Fx, Fy, Fz, p[i].vx, p[i].vy,
  // p[i].vz are updated in the outer loop thus, we only parallelize the outer
  // loop
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
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

__global__ void bodyForceKernel2(Body_t *p, float dt, int n) {
  // We can't parallelize both i and j, since Fx, Fy, Fz, p[i].vx, p[i].vy,
  // p[i].vz are updated in the outer loop thus, we only parallelize the outer
  // loop
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ Body_t p_tile[BLOCKSIZE];
  Body_t currBody;
  if (i < n) {
    currBody = p[i];
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    for (int tile = 0; tile < (n - 1) / BLOCKSIZE + 1; tile++) {
      if (tile * BLOCKSIZE + threadIdx.x < n)
        p_tile[threadIdx.x] = p[tile * BLOCKSIZE + threadIdx.x];
      __syncthreads();

      for (int j = 0; j < n; j++) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
      }
    }
    __syncthreads();
    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
  }
}

__global__ void updateBodyPositionsKernel(Body_t *p, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
  }
}

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body_t *p, float dt, int n) {
  for (int i = 0; i < n; i++) {
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
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

void updateBodyPositions(Body_t *p, float dt, int n) {
  for (int i = 0; i < n; i++) {
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
  }
}

int main(const int argc, const char **argv) {
  int nBodies = 30000;
  if (argc > 1)
    nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  double cpu_start, cpu_end;

  Body_t *p = (Body_t *)malloc(nBodies * sizeof(Body_t));

  Body_t *p_d;

  CHECK(cudaMalloc((void **)&p_d, sizeof(Body_t) * nBodies));

  randomizeBodies((float *)p, 6 * nBodies); // Init position / speed data

  CHECK(cudaMemcpy(p_d, p, sizeof(Body_t) * nBodies, cudaMemcpyHostToDevice));
  dim3 threadsPerBlock(BLOCKSIZE, 1, 1);
  dim3 blocksPerGrid((nBodies - 1) / BLOCKSIZE + 1, 1, 1);
  for (int iter = 1; iter <= nIters; iter++) {
    bodyForceKernel<<<blocksPerGrid, threadsPerBlock>>>(p_d, dt, nBodies);
    CHECK_KERNELCALL();
    updateBodyPositionsKernel<<<blocksPerGrid, threadsPerBlock>>>(p_d, dt,
                                                                  nBodies);
    CHECK_KERNELCALL()
  }

  CHECK(cudaFree(p_d));

  cpu_start = get_time();
  for (int iter = 1; iter <= nIters; iter++) {
    bodyForce(p, dt, nBodies);           // compute interbody forces
    updateBodyPositions(p, dt, nBodies); // integrate position
  }
  cpu_end = get_time();
  double avgTime = (cpu_end - cpu_start) / nIters;

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies,
         1e-9 * nBodies * nBodies / avgTime);
  free(p);

  return 0;
}

// function to get the time of day in seconds
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
