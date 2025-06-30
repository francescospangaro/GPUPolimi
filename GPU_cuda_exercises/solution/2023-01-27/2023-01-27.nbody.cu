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

#define SOFTENING 1e-9f
#define BLOCK_SIZE 32
#define TRESHOLD 1e-5

typedef struct
{
  double x, y, z, vx, vy, vz;
} Body_t;

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
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
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

__global__ void bodyForce_gpu_opt(Body_t *p, double dt, int n)
{
  __shared__ Body_t p_shared[BLOCK_SIZE];
  if (i < n)
  {
  }
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
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

__global__ void updateBodyPositions_gpu(Body_t *p, double dt, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
  }
}

void updateBodyPositions(Body_t *p, double dt, int n)
{
  for (int i = 0; i < n; i++)
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

  const double dt = 0.01f; // time step
  const int nIters = 10;   // simulation iterations

  Body_t *p = (Body_t *)malloc(nBodies * sizeof(Body_t));

  randomizeBodies((double *)p, 6 * nBodies); // Init position / speed data

  Body_t *d_p;
  Body_t *out_p = (Body_t *)malloc(nBodies * sizeof(Body_t));

  CHECK(cudaMalloc(&d_p, sizeof(Body_t) * nBodies));
  CHECK(cudaMemcpy(d_p, p, sizeof(Body_t) * nBodies, cudaMemcpyHostToDevice));

  for (int iter = 1; iter <= nIters; iter++)
  {
    // bodyForce(p, dt, nBodies);           // compute interbody forces
    // updateBodyPositions(p, dt, nBodies); // integrate position

    bodyForce_gpu<<<(nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_p, dt, nBodies); // compute interbody forces
    CHECK_KERNELCALL();
    updateBodyPositions_gpu<<<(nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_p, dt, nBodies); // integrate position
    CHECK_KERNELCALL();
  }

  CHECK(cudaMemcpy(out_p, d_p, sizeof(Body_t) * nBodies, cudaMemcpyDeviceToHost));

  /*for (int i = 0; i < nBodies; i++)
  {
    if (std::abs(p[i].x - out_p[i].x) > TRESHOLD || std::abs(p[i].y - out_p[i].y) > TRESHOLD || std::abs(p[i].z - out_p[i].z) > TRESHOLD ||
        std::abs(p[i].vx - out_p[i].vx) > TRESHOLD || std::abs(p[i].vy - out_p[i].vy) > TRESHOLD || std::abs(p[i].vz - out_p[i].vz) > TRESHOLD)
    {
      printf("ERROR %.10f, %.10f, %.10f    %.10f, %.10f, %.10f\n", p[i].x, p[i].y, p[i].z, out_p[i].x, out_p[i].y, out_p[i].z);
      printf("ERROR %.10f, %.10f, %.10f    %.10f, %.10f, %.10f\n", p[i].vx, p[i].vy, p[i].vz, out_p[i].vx, out_p[i].vy, out_p[i].vz);
      return 1;
    }
  }*/
  printf("ok");

  free(p);
  CHECK(cudaFree(d_p));

  return 0;
}
