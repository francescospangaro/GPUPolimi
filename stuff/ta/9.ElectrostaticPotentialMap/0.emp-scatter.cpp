#include <cstdlib>
#include <math.h>
#include <random>
#include <vector>

#define GRID_DIM     128
#define NUM_ATOMS    64
#define ATOM_STRIDE  4   // Field in the atom data structure
#define GRID_SPACING 0.5 //Angstrom

#define MIN_COORD (GRID_SPACING * GRID_DIM)
#define MAX_COORD (GRID_SPACING * GRID_DIM)

#define MIN_CHARGE -5.f
#define MAX_CHARGE 5.f

void cenergy(float *energygrid, const float *atoms) {
  for (int t = 0; t < GRID_DIM; t++) {
    const float z = GRID_SPACING * (float) t;
    for (int j = 0; j < GRID_DIM; j++) {
      // calculate y coordinate of the grid point based on j
      float y = GRID_SPACING * (float) j;
      for (int i = 0; i < GRID_DIM; i++) {
        // calculate x coordinate based on i
        float x      = GRID_SPACING * (float) i;
        float energy = 0.0f;
        for (int n = 0; n < NUM_ATOMS * ATOM_STRIDE; n += ATOM_STRIDE) {
          float dx = x - atoms[n];
          float dy = y - atoms[n + 1];
          float dz = z - atoms[n + 2];
          energy += atoms[n + 3] / sqrtf(dx * dx + dy * dy + dz * dz);
        }
        energygrid[GRID_DIM * GRID_DIM * t + GRID_DIM * j + i] = energy;
      }
    }
  }
}

int main() {
  std::vector<float> atoms(NUM_ATOMS * ATOM_STRIDE);
  std::vector<float> energygrid(GRID_DIM * GRID_DIM * GRID_DIM, 0.f);

  for (int i = 0; i < NUM_ATOMS * ATOM_STRIDE; i += ATOM_STRIDE) {
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