/*
 * The program simulates the heat transfer in a 2D surface. The 2D surface is modeled by
 * means of a 2D array called surf (linearized for the sake of simplicity); each position in
 * the array surf[i][j] contains the temperature of a single portion of the surface.
 * Heat transfer is simulated by a partial differential equation (not discussed here in
 * details...). The equation is solved by iterating over nsteps subsequent time steps; at
 * each time step n, the new temperature of each position is computed as a function of the
 * current temperatures of the same position surf[i][j] and of the 4 neighbor positions
 * surf[i-1][j], surf[i][j-1], surf[i+1][j] and surf[i][j+1]; results are saved in a new
 * array surf1. surf1 and surf are then swapped to simulate subsequent time step (surf1
 * contains the input of the subsequent step while surf is used to store the results).
 * When computing the temperature of a position on the border of the surface, the
 * non-existing neighbor positions are replaced with the temperature of the central position
 * itself (i.e., when computing the temperature surf1[0][2], the neighbor position
 * surf[-1][2] is replaced with surf[0][2]).
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

#define BLOCKDIM 32
#define COARSENING_FACTOR_X 8
#define EPS 0.01f

#define A 0.5f   // Diffusion constant
#define DX 0.01f // Horizontal grid spacing
#define DY 0.01f // Vertical grid spacing
#define DX2 (DX * DX)
#define DY2 (DY * DY)
#define DT (DX2 * DY2 / (2.0f * A * (DX2 + DY2))) // Largest stable time step
#define SMALLSIZEPROBLEM 100

void print(float *m, int nx, int ny);
__device__ __host__ int getIndex(int const i, int const j, int const width);

void init(float *const surf,
          int const nx,
          int const ny)
{
  // Initialize the data with a pattern of disk of radius of 1/6 of the width
  float radius2 = (nx / 6.0f) * (nx / 6.0f);
  for (int i = 0; i < nx; i++)
  {
    for (int j = 0; j < ny; j++)
    {
      int index = getIndex(i, j, ny);
      // Distance of point i, j from the origin
      float ds2 = (i - nx / 2) * (i - nx / 2) + (j - ny / 2) * (j - ny / 2);
      if (ds2 < radius2)
        surf[index] = 65.0f;
      else
        surf[index] = 5.0f;
    }
  }
}

void compute(float *surf,
             float *surf1,
             int const nx,
             int const ny,
             int const nsteps)
{
  // Compute-intensive kernel for the heat simulation
  int index;
  float sij, sim1j, sijm1, sip1j, sijp1;

  // Simulate N time steps
  for (int n = 0; n < nsteps; n++)
  {
    // Go through the entire 2D surface
    for (int i = 0; i < nx; i++)
    {
      for (int j = 0; j < ny; j++)
      {
        // Compute the heat transfer taking old temperature from surf
        // and saving new temperature in surf1
        index = getIndex(i, j, ny);
        sij = surf[index];
        if (i > 0)
          sim1j = surf[getIndex(i - 1, j, ny)];
        else
          sim1j = sij;
        if (j > 0)
          sijm1 = surf[getIndex(i, j - 1, ny)];
        else
          sijm1 = sij;
        if (i < nx - 1)
          sip1j = surf[getIndex(i + 1, j, ny)];
        else
          sip1j = sij;
        if (j < ny - 1)
          sijp1 = surf[getIndex(i, j + 1, ny)];
        else
          sijp1 = sij;
        surf1[index] = sij + A * DT * ((sim1j - 2.0 * sij + sip1j) / DX2 + (sijm1 - 2.0 * sij + sijp1) / DY2);
      }
    }
    // Swap the surf and surf1 pointers for the next time step
    // (it avoids copying all data from surf1 to surf)
    float *tmp = surf;
    surf = surf1;
    surf1 = tmp;
  }
}

__global__ void initialize(float *const __restrict__ surf,
                           int const nx,
                           int const ny)
{
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  int const j = blockIdx.y * blockDim.y + threadIdx.y;
  // Initialize the data with a pattern of disk of radius of 1/6 of the width
  float radius2 = (nx / 6.0f) * (nx / 6.0f);
  if (i < nx && j < ny)
  {
    int index = getIndex(i, j, ny);
    // Distance of point i, j from the origin
    float ds2 = (i - nx / 2) * (i - nx / 2) + (j - ny / 2) * (j - ny / 2);
    if (ds2 < radius2)
      surf[index] = 65.0f;
    else
      surf[index] = 5.0f;
  }
}

__global__ void initializeCoarsening(float *const __restrict__ surf,
                                     int const nx,
                                     int const ny)
{
  int const i0 = blockIdx.x * blockDim.x + threadIdx.x;
  int const j = blockIdx.y * blockDim.y + threadIdx.y;
  // Initialize the data with a pattern of disk of radius of 1/6 of the width
  float radius2 = (nx / 6.0f) * (nx / 6.0f);
  if (j < ny)
  {
    for (int i = i0; i < nx; i += blockDim.x * gridDim.x)
    {
      int index = getIndex(i, j, ny);
      // Distance of point i, j from the origin
      float ds2 = (i - nx / 2) * (i - nx / 2) + (j - ny / 2) * (j - ny / 2);
      if (ds2 < radius2)
        surf[index] = 65.0f;
      else
        surf[index] = 5.0f;
    }
  }
}

__global__ void computeStep(float *const __restrict__ surf,
                            float *const __restrict__ surf1,
                            int const nx,
                            int const ny)
{
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  int const j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nx && j < ny)
  {
    int index;
    float sij, sim1j, sijm1, sip1j, sijp1;

    // Compute the heat transfer taking old temperature from surf
    // and saving new temperature in surf1
    index = getIndex(i, j, ny);
    sij = surf[index];
    if (i > 0)
      sim1j = surf[getIndex(i - 1, j, ny)];
    else
      sim1j = sij;
    if (j > 0)
      sijm1 = surf[getIndex(i, j - 1, ny)];
    else
      sijm1 = sij;
    if (i < nx - 1)
      sip1j = surf[getIndex(i + 1, j, ny)];
    else
      sip1j = sij;
    if (j < ny - 1)
      sijp1 = surf[getIndex(i, j + 1, ny)];
    else
      sijp1 = sij;
    surf1[index] = sij + A * DT * ((sim1j - 2.0 * sij + sip1j) / DX2 + (sijm1 - 2.0 * sij + sijp1) / DY2);
  }
}

bool arraySimilar(float *arr1, float *arr2, int n)
{
  for (int i = 0; i < n; ++i)
    if (abs(arr1[i] - arr2[i]) > EPS)
    {
      printf("Diff: %f\n", arr1[i] - arr2[i]);
      return false;
    }
  return true;
}

int main(int argc, char **argv)
{
  float *surf, *surf1;
  int nx, ny, nsteps;

  // Read arguments
  if (argc != 4 || atoi(argv[3]) % 2 != 0)
  {
    printf("Please specify sizes of the 2D surface and the number of time steps\n");
    printf("The number of time steps must be even\n");
    return 0;
  }
  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  nsteps = atoi(argv[3]);

  // Allocate memory for the two arrays
  surf = (float *)malloc(sizeof(float) * nx * ny);
  if (!surf)
  {
    printf("Error: malloc failed\n");
    return 1;
  }
  surf1 = (float *)malloc(sizeof(float) * nx * ny);
  if (!surf1)
  {
    printf("Error: malloc failed\n");
    return 1;
  }

  init(surf, nx, ny);

  // Print initial temperatures only for small problems
  if (nx * ny <= SMALLSIZEPROBLEM)
    print(surf, nx, ny);

  compute(surf, surf1, nx, ny, nsteps);

  // Print final temperatures only for small problems
  if (nx * ny <= SMALLSIZEPROBLEM)
  {
    printf("\n\n");
    print(surf, nx, ny);
  }

  // Do the same on the GPU
  float *surf_d, *surf1_d, *surf_h;

  surf_h = (float *)malloc(sizeof(float) * nx * ny);
  if (!surf_h)
  {
    printf("Error: malloc failed\n");
    return 1;
  }

  CHECK(cudaMalloc(&surf_d, sizeof(float) * nx * ny));
  CHECK(cudaMalloc(&surf1_d, sizeof(float) * nx * ny));

  dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
  dim3 numOfBlocks((nx + BLOCKDIM - 1) / BLOCKDIM, (ny + BLOCKDIM - 1) / BLOCKDIM);
  dim3 numOfBlocksCoarsened(
      (nx + (BLOCKDIM * COARSENING_FACTOR_X) - 1) / (BLOCKDIM * COARSENING_FACTOR_X),
      (ny + BLOCKDIM - 1) / BLOCKDIM);

  // Kernel 1
  initialize<<<numOfBlocks, threadsPerBlock>>>(surf_d, nx, ny);
  CHECK_KERNELCALL();

  for (int n = 0; n < nsteps; n++)
  {
    computeStep<<<numOfBlocks, threadsPerBlock>>>(surf_d, surf1_d, nx, ny);
    CHECK_KERNELCALL();

    // Swap the surf and surf1 pointers for the next time step
    // (it avoids copying all data from surf1 to surf)
    float *tmp = surf_d;
    surf_d = surf1_d;
    surf1_d = tmp;
  }

  CHECK(cudaMemcpy(surf_h, surf_d, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Kernel 1: %s\n", arraySimilar(surf, surf_h, nx * ny) ? "OK" : "KO");

  // Kernel 2
  initializeCoarsening<<<numOfBlocksCoarsened, threadsPerBlock>>>(surf_d, nx, ny);
  CHECK_KERNELCALL();

  for (int n = 0; n < nsteps; n++)
  {
    computeStep<<<numOfBlocks, threadsPerBlock>>>(surf_d, surf1_d, nx, ny);
    CHECK_KERNELCALL();

    // Swap the surf and surf1 pointers for the next time step
    // (it avoids copying all data from surf1 to surf)
    float *tmp = surf_d;
    surf_d = surf1_d;
    surf1_d = tmp;
  }

  CHECK(cudaMemcpy(surf_h, surf_d, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Kernel 2: %s\n", arraySimilar(surf, surf_h, nx * ny) ? "OK" : "KO");

  // Cleanup

  cudaFree(surf_d);
  cudaFree(surf1_d);

  // Free memory
  free(surf);
  free(surf1);

  return 0;
}

// Take in input 2D coordinates of a point in a matrix
// and translate in a 1D offset for the linearized matrix
int getIndex(int const i, int const j, int const width)
{
  return (i * width + j);
}

// Display a vector of numbers on the screen
void print(float *m, int nx, int ny)
{
  int i, j;
  for (i = 0; i < nx; i++)
  {
    for (j = 0; j < ny; j++)
      printf("%6.3f ", m[getIndex(i, j, ny)]);
    printf("\n");
  }
}
