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
#include <sys/time.h>

#define SOFTENING 1e-9f
#define BLOCK_DIM 32
#define TRESHOLD 1e-2
#define TILE_DIM BLOCK_DIM

typedef struct
{
  double x, y, z, vx, vy, vz;
} Body_t;

double get_time();
void randomizeBodies(double *data, int n);
void bodyForce(Body_t *p, double dt, int n);
void updateBodyPositions(Body_t *p, double dt, int n);

void randomizeBodies(double *data, int n)
{
  for (int i = 0; i < n; i++)
  {
    data[i] = 2.0f * (rand() / (double)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body_t *p, double dt, int n)
{
  for (int i = 0; i < n; i++)
  {
    double Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int j = 0; j < n; j++)
    {
      double dx = p[j].x - p[i].x;
      double dy = p[j].y - p[i].y;
      double dz = p[j].z - p[i].z;
      double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      double invDist = 1.0f / sqrtf(distSqr);
      double invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
  }
}

__global__ void bodyForce_gpu(Body_t *p, double dt, int n)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n)
  {
    double Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    for (int j = 0; j < n; j++)
    {
      double dx = p[j].x - p[tid].x;
      double dy = p[j].y - p[tid].y;
      double dz = p[j].z - p[tid].z;
      double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      double invDist = 1.0f / sqrtf(distSqr);
      double invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    p[tid].vx += dt * Fx;
    p[tid].vy += dt * Fy;
    p[tid].vz += dt * Fz;
  }
}

/*__global__ void bodyForce_shared_gpu(Body_t *p, double dt, int n)
{
  __shared__ Body_t tile_s[TILE_DIM];
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < n)
  {
    for (int iter = 0; iter < (n - 1) / TILE_DIM + 1; iter++)
    {
      for (int k = threadIdx.x; k < TILE_DIM; k += BLOCK_DIM)
      {
        tile_s[threadIdx.x] = p[iter * TILE_DIM + threadIdx.x];
      }
      __syncthreads();
      double Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
      for (int j = 0; j < TILE_DIM; j++)
      {
        double dx = tile_s[j].x - p[tid].x;
        double dy = tile_s[j].y - p[tid].y;
        double dz = tile_s[j].z - p[tid].z;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = 1.0f / sqrtf(distSqr);
        double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
      }

      p[tid].vx += dt * Fx;
      p[tid].vy += dt * Fy;
      p[tid].vz += dt * Fz;
      __syncthreads();
    }
  }
}*/

void updateBodyPositions(Body_t *p, double dt, int n)
{
  for (int i = 0; i < n; i++)
  {
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
  }
}

__global__ void updateBodyPositions_gpu(Body_t *p, double dt, int n)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n)
  {
    p[tid].x += p[tid].vx * dt;
    p[tid].y += p[tid].vy * dt;
    p[tid].z += p[tid].vz * dt;
  }
}

int main(const int argc, const char **argv)
{
  int nBodies = 30000;
  if (argc > 1)
    nBodies = atoi(argv[1]);

  const double dt = 0.01f; // time step
  const int nIters = 10;   // simulation iterations
  // double cpu_start, cpu_end;

  Body_t *p = (Body_t *)malloc(nBodies * sizeof(Body_t));

  randomizeBodies((double *)p, 6 * nBodies); // Init position / speed data

  Body_t *d_p;

  cudaMalloc(&d_p, sizeof(Body_t) * nBodies);
  cudaMemcpy(d_p, p, sizeof(Body_t) * nBodies, cudaMemcpyHostToDevice);

  // cpu_start = get_time();
  for (int iter = 1; iter <= nIters; iter++)
  {
    bodyForce(p, dt, nBodies);           // compute interbody forces
    updateBodyPositions(p, dt, nBodies); // integrate position
  }
  // cpu_end = get_time();
  // double avgTime = (cpu_end - cpu_start) / nIters;

  printf("CPU res:\n");
  // printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

  dim3 threadsPerBlock(BLOCK_DIM);
  dim3 blocksPerGrid((nBodies - 1) / BLOCK_DIM + 1);

  for (int iter = 1; iter <= nIters; iter++)
  {
    bodyForce_shared_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_p, dt, nBodies);
    updateBodyPositions_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_p, dt, nBodies);
  }
  cudaDeviceSynchronize();

  Body_t *gpu_res = (Body_t *)malloc(nBodies * sizeof(Body_t));
  cudaMemcpy(gpu_res, d_p, sizeof(Body_t) * nBodies, cudaMemcpyDeviceToHost);

  for (int i = 0; i < nBodies; i++)
  {
    if (std::abs(p[i].x - gpu_res[i].x) > TRESHOLD || std::abs(p[i].y - gpu_res[i].y) > TRESHOLD || std::abs(p[i].z - gpu_res[i].z) > TRESHOLD ||
        std::abs(p[i].vx - gpu_res[i].vx) > TRESHOLD || std::abs(p[i].vy - gpu_res[i].vy) > TRESHOLD || std::abs(p[i].vz - gpu_res[i].vz) > TRESHOLD)
    {
      printf("ERROR %.10f, %.10f, %.10f    %.10f, %.10f, %.10f\n", p[i].x, p[i].y, p[i].z, gpu_res[i].x, gpu_res[i].y, gpu_res[i].z);
      printf("ERROR %.10f, %.10f, %.10f    %.10f, %.10f, %.10f\n", p[i].vx, p[i].vy, p[i].vz, gpu_res[i].vx, gpu_res[i].vy, gpu_res[i].vz);
      return 1;
    }
  }
  printf("ALL OK...\n");

  cudaDeviceReset();
  free(p);

  return 0;
}

// function to get the time of day in seconds
double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
