#include <cstdlib>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>

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

#define GRID_DIM 128
#define NUM_ATOMS 64
#define ATOM_STRIDE 4    // Field in the atom data structure
#define GRID_SPACING 0.5 // Angstrom

#define MIN_COORD 0
#define MAX_COORD (GRID_SPACING * GRID_DIM)

#define MIN_CHARGE -5.f
#define MAX_CHARGE 5.f

#define BLOCK_DIM 128

void cenergy(float *energygrid, const float *atoms)
{
  for (int t = 0; t < GRID_DIM; t++)
  {
    const float z = GRID_SPACING * (float)t;
    // starting point of the slice in the energy grid
    const int grid_slice_offset = (GRID_DIM * GRID_DIM * z) / GRID_SPACING;
    // calculate potential contribution of each atom
    for (int n = 0; n < NUM_ATOMS * ATOM_STRIDE; n += ATOM_STRIDE)
    {
      const float dz = z - atoms[n + 2]; // all grid points in a slice have the same
      const float dz2 = dz * dz;         // z value, no recalculation in inner loops
      const float charge = atoms[n + 3];
      for (int j = 0; j < GRID_DIM; j++)
      {
        const float y = GRID_SPACING * (float)j;
        const float dy = y - atoms[n + 1]; // all grid points in a row have the same
        const float dy2 = dy * dy;         // y value
        const int grid_row_offset = grid_slice_offset + GRID_DIM * j;
        for (int i = 0; i < GRID_DIM; i++)
        {
          const float x = GRID_SPACING * (float)i;
          const float dx = x - atoms[n];
          energygrid[grid_row_offset + i] += charge / sqrtf(dx * dx + dy2 + dz2);
        }
      }
    }
  }
}

__constant__ float d_atoms[NUM_ATOMS * ATOM_STRIDE];

void __global__ cenergy_gpu(float *energygrid)
{
  const int n = (blockIdx.x * blockDim.x + threadIdx.x) * ATOM_STRIDE;
  for (int t = 0; t < GRID_DIM; t++)
  {
    const float z = GRID_SPACING * (float)t;
    // TODO
    const float dz = z - d_atoms[n + 2]; // all grid points in a slice have the same
    const float dz2 = dz * dz;           // z value, no recalculation in inner loops
    // starting point of the slice in the energy grid
    const int grid_slice_offset = (GRID_DIM * GRID_DIM * z) / GRID_SPACING;
    const float charge = d_atoms[n + 3];
    for (int j = 0; j < GRID_DIM; j++)
    {
      const float y = GRID_SPACING * (float)j;
      const float dy = y - d_atoms[n + 1]; // all grid points in a row have the same
      const float dy2 = dy * dy;           // y value
      const int grid_row_offset = grid_slice_offset + GRID_DIM * j;
      for (int i = 0; i < GRID_DIM; i++)
      {
        const float x = GRID_SPACING * (float)i;
        const float dx = x - d_atoms[n];
        atomicAdd(&energygrid[grid_row_offset + i], charge / sqrtf(dx * dx + dy2 + dz2));
      }
    }
    __syncthreads();
  }
}

int main()
{
  std::vector<float> atoms(NUM_ATOMS * ATOM_STRIDE);
  std::vector<float> energygrid(GRID_DIM * GRID_DIM * GRID_DIM, 0.f);

  for (int i = 0; i < NUM_ATOMS * ATOM_STRIDE; i += ATOM_STRIDE)
  {
    atoms[i + 0] =
        MIN_COORD + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (MAX_COORD - MIN_COORD)));
    atoms[i + 1] =
        MIN_COORD + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (MAX_COORD - MIN_COORD)));
    atoms[i + 2] =
        MIN_COORD + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (MAX_COORD - MIN_COORD)));
    atoms[i + 3] =
        MIN_CHARGE + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (MIN_CHARGE - MAX_CHARGE)));
  }

  // Launch CPU version
  cenergy(energygrid.data(), atoms.data());

  float *d_energygrid;
  CHECK(cudaMalloc(&d_energygrid, sizeof(float) * GRID_DIM * GRID_DIM * GRID_DIM));
  CHECK(cudaMemset(d_energygrid, 0, sizeof(float) * GRID_DIM * GRID_DIM * GRID_DIM));
  CHECK(cudaMemcpyToSymbol(d_atoms, atoms.data(), sizeof(float) * NUM_ATOMS * ATOM_STRIDE));

  const dim3 threadsPerBlock(BLOCK_DIM);
  const dim3 numBlocks((NUM_ATOMS + BLOCK_DIM - 1) / BLOCK_DIM);
  cenergy_gpu<<<numBlocks, threadsPerBlock>>>(d_energygrid);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  std::vector<float> h_energygrid(GRID_DIM * GRID_DIM * GRID_DIM);
  CHECK(cudaMemcpy(h_energygrid.data(),
                   d_energygrid,
                   sizeof(float) * GRID_DIM * GRID_DIM * GRID_DIM,
                   cudaMemcpyDeviceToHost));

  for (int t = 0; t < GRID_DIM; t++)
  {
    for (int j = 0; j < GRID_DIM; j++)
    {
      for (int i = 0; i < GRID_DIM; i++)
      {
        if (std::abs(energygrid[GRID_DIM * GRID_DIM * t + GRID_DIM * j + i] -
                     h_energygrid[GRID_DIM * GRID_DIM * t + GRID_DIM * j + i]) > 1e-3)
        {
          std::cout << "Electrostatic Map Potential CPU and GPU are NOT equivalent!" << std::endl;
          return EXIT_FAILURE;
        }
      }
    }
  }
  std::cout << "Electrostatic Map Potential CPU and GPU are equivalent!" << std::endl;

  return EXIT_SUCCESS;
}