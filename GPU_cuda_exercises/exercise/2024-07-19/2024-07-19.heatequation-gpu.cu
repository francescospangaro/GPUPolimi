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

#define A 0.5f   // Diffusion constant
#define DX 0.01f // Horizontal grid spacing
#define DY 0.01f // Vertical grid spacing
#define DX2 (DX * DX)
#define DY2 (DY * DY)
#define DT (DX2 * DY2 / (2.0f * A * (DX2 + DY2))) // Largest stable time step
#define SMALLSIZEPROBLEM 100
#define COARSENING_FACTOR 8
#define BLOCK_DIM 32

void print(float *m, int nx, int ny);
int getIndex(int i, int j, int width);

__global__ void initialize(float *surf,
                           const unsigned int dimRow,
                           const unsigned int dimCol,
                           const float radius2)
{
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;

  if (tx < dimCol && ty < dimRow)
  {
    int index = getIndex(tx, ty, dimRow);
    float ds2 = (tx - dimCol / 2) * (tx - dimCol / 2) + (ty - dimRow / 2) * (ty - dimRow / 2);
    if (ds2 < radius2)
      surf[index] = 65.0f;
    else
      surf[index] = 5.0f;
  }
}

__global__ void simulateStep(
    float *currSurf,
    float *nextSurf,
    const unsigned int dimRow,
    const unsigned int dimCol)
{
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;

  int index;
  float sij, sim1j, sijm1, sip1j, sijp1;
  if (tx < dimCol && ty < dimRow)
  {
    index = getIndex(tx, ty, dimRow);
    sij = currSurf[index];
    if (tx > 0)
      sim1j = currSurf[getIndex(tx - 1, ty, dimRow)];
    else
      sim1j = sij;
    if (ty > 0)
      sijm1 = currSurf[getIndex(tx, ty - 1, dimRow)];
    else
      sijm1 = sij;
    if (tx < dimCol - 1)
      sip1j = currSurf[getIndex(tx + 1, ty, dimRow)];
    else
      sip1j = sij;
    if (ty < dimRow - 1)
      sijp1 = currSurf[getIndex(tx, ty + 1, dimRow)];
    else
      sijp1 = sij;
    nextSurf[index] = sij + A * DT *
                                ((sim1j - 2.0 * sij + sip1j) / DX2 + (sijm1 - 2.0 * sij + sijp1) / DY2);
  }
}

__global__ void initialize_coarsening(float *surf,
                                      const unsigned int dimRow,
                                      const unsigned int dimCol,
                                      const float radius2)
{
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;

  int stride = blockDim.x * gridDim.x * COARSENING_FACTOR;
  if (ty < dimRow)
  {
    for (int i = tx; i < dimCol; i += stride)
    {
      int index = getIndex(i, ty, dimRow);
      float ds2 = (i - dimCol / 2) * (i - dimCol / 2) + (ty - dimRow / 2) * (ty - dimRow / 2);
      if (ds2 < radius2)
        surf[index] = 65.0f;
      else
        surf[index] = 5.0f;
    }
  }
}

// In the main function we must adapt the number of threads spawned by the kernel function
// In fact in the non coarsened solution we could spawn 32x32 blocks and adapt the number of
// blocks to the number of data to process, with this new kernel we have also to keep in regard
// the COARSENING_FACTOR, therefore the main will contain:

//  dim3 blocksPerGrid((nx-1)/(threadsPerBlock.x*COARSENING_FACTOR)+1, (nx-1)/(threadsPerBlock.y*COARSENING_FACTOR)+1);

int main(int argc, char **argv)
{
  float *surf, *surf1, *tmp;
  int nx, ny, nsteps, n;

  // Read arguments
  //  if (argc != 4 || atoi(argv[3]) % 2 != 0)
  //  {
  //   printf("Please specify sizes of the 2D surface and the number of time steps\n");
  //   printf("The number of time steps must be even\n");
  //   return 0;
  //  }
  //  nx = atoi(argv[1]);
  //  ny = atoi(argv[2]);
  //  nsteps = atoi(argv[3]);
  nx = 8;
  ny = 10;
  nsteps = 6;

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

  float *d_surf, *d_surf1;
  cudaMalloc(&d_surf, sizeof(float) * nx * ny);
  cudaMalloc(&d_surf1, sizeof(float) * nx * ny);
  // Initialize the data with a pattern of disk of radius of 1/6 of the width
  float radius2 = (nx / 6.0f) * (nx / 6.0f);

  dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
  dim3 blocksPerGrid((nx - 1) / (threadsPerBlock.x * COARSENING_FACTOR) + 1, (ny - 1) / threadsPerBlock.y + 1);
  initialize_coarsening<<<blocksPerGrid, threadsPerBlock>>>(d_surf, ny, nx, radius2);
  cudaDeviceSynchronize();

  cudaMemcpy(surf, d_surf, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);

  // for (i = 0; i < nx; i++)
  // {
  //   for (j = 0; j < ny; j++)
  //   {
  //     int index = getIndex(i, j, ny);
  //     // Distance of point i, j from the origin
  //     float ds2 = (i - nx / 2) * (i - nx / 2) + (j - ny / 2) * (j - ny / 2);
  //     if (ds2 < radius2)
  //       surf[index] = 65.0f;
  //     else
  //       surf[index] = 5.0f;
  //   }
  // }

  // Print initial temperatures only for small problems
  if (nx * ny <= SMALLSIZEPROBLEM)
    print(surf, nx, ny);

  // Compute-intensive kernel for the heat simulation
  // int index;
  // float sij, sim1j, sijm1, sip1j, sijp1;

  cudaMalloc(&d_surf1, sizeof(float) * nx * ny);
  // Simulate N time steps
  for (n = 0; n < nsteps; n++)
  {
    if (n != 0)
      cudaMemcpy(d_surf, surf, sizeof(float) * nx * ny, cudaMemcpyHostToDevice);
    // Go through the entire 2D surface
    // for (i = 0; i < nx; i++)
    // {
    //   for (int j = 0; j < ny; j++)
    //   {
    //     // Compute the heat transfer taking old temperature from surf
    //     // and saving new temperature in surf1
    //     index = getIndex(i, j, ny);
    //     sij = surf[index];
    //     if (i > 0)
    //       sim1j = surf[getIndex(i - 1, j, ny)];
    //     else
    //       sim1j = sij;
    //     if (j > 0)
    //       sijm1 = surf[getIndex(i, j - 1, ny)];
    //     else
    //       sijm1 = sij;
    //     if (i < nx - 1)
    //       sip1j = surf[getIndex(i + 1, j, ny)];
    //     else
    //       sip1j = sij;
    //     if (j < ny - 1)
    //       sijp1 = surf[getIndex(i, j + 1, ny)];
    //     else
    //       sijp1 = sij;
    //     surf1[index] = sij + A * DT *
    //                              ((sim1j - 2.0 * sij + sip1j) / DX2 + (sijm1 - 2.0 * sij + sijp1) / DY2);
    //   }
    // }
    simulateStep<<<blocksPerGrid, threadsPerBlock>>>(d_surf, d_surf1, ny, nx);
    cudaDeviceSynchronize();

    cudaMemcpy(surf1, d_surf1, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);
    // Swap the surf and surf1 pointers for the next time step
    // (it avoids copying all data from surf1 to surf)
    tmp = surf;
    surf = surf1;
    surf1 = tmp;
  }

  // Print final temperatures only for small problems
  if (nx * ny <= SMALLSIZEPROBLEM)
  {
    printf("\n\n");
    print(surf, nx, ny);
  }
  cudaDeviceReset();
  // Free memory
  free(surf);
  free(surf1);

  return 0;
}

// Take in input 2D coordinates of a point in a matrix
// and translate in a 1D offset for the linearized matrix
__host__ __device__ int getIndex(int i, int j, int width)
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
