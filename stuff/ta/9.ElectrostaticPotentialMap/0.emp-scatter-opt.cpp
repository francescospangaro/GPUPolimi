#include <cstdlib>
#include <math.h>
#include <random>
#include <vector>

#define GRID_DIM 128
#define NUM_ATOMS 64
#define ATOM_STRIDE 4    // Field in the atom data structure
#define GRID_SPACING 0.5 // Angstrom

#define MIN_COORD (GRID_SPACING * GRID_DIM)
#define MAX_COORD (GRID_SPACING * GRID_DIM)

#define MIN_CHARGE -5.f
#define MAX_CHARGE 5.f

void cenergy(float *energygrid, const float *atoms)
{
  for (int t = 0; t < GRID_DIM; t++)
  {
    const float z = GRID_SPACING * (float)t;
    // Starting point of the slice in the energy grid
    const int grid_slice_offset = (GRID_DIM * GRID_DIM * z) / GRID_SPACING;
    // Calculate potential contribution of each atom
    for (int n = 0; n < NUM_ATOMS * ATOM_STRIDE; n += ATOM_STRIDE)
    {
      const float dz = z - atoms[n + 2]; // All grid points in a slice have the same
      const float dz2 = dz * dz;         // z value, no recalculation in inner loops
      const float charge = atoms[n + 3];
      for (int j = 0; j < GRID_DIM; j++)
      {
        const float y = GRID_SPACING * (float)j;
        const float dy = y - atoms[n + 1]; // All grid points in a row have the same y value
        const float dy2 = dy * dy;
        const int grid_row_offset = grid_slice_offset + GRID_DIM * j;
        for (int i = 0; i < GRID_DIM; i++)
        {
          float x = GRID_SPACING * (float)i;
          float energy = 0.0f;
          float dx = x - atoms[n];
          energygrid[grid_row_offset + i] = charge / sqrtf(dx * dx + dy2 + dz2);
        }
      }
    }
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

  return EXIT_SUCCESS;
}