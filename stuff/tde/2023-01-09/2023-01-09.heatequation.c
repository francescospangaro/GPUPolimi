/*
 * The program simulates the heat transfer in a 2D surface. The 2D surface is modeled by
 * means of a 2D array called surf (linearized for the sake of simplicity); each position in
 * the array surf[i][j] contains the temperature of a single portion of the surface.
 * Heat transfer is simulated by a partial differential equation (not discussed here in
 * details...). The equation is solved by iterating over nsteps subsequent time steps; at
 * each time step n, the new temperature of each position is computed as a function of the
 * current temperatures of the same position surf[i][j] and of the 4 neighbor positions
 * surf[i-1][j], surf[i][j-1], surf[i+1][j] and surf[i][j+1]; results are saved in a new
 * array surf1, which is then swapped with surf to simulate subsequent time step.
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
#define PROBLEMAPICCOLO 100

void print(float *m, int nx, int ny);
int getIndex(int i, int j, int width);

int main(int argc, char **argv)
{
  float *surf, *surf1, *tmp;
  int i, j, nx, ny, nsteps, n;

  // Read arguments
  if (argc != 4)
  {
    printf("Please specify sizes of the 2D surface and the number of time steps\n");
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

  // Initialize the data with a pattern of disk of radius of 1/6 of the width
  float radius2 = (nx / 6.0f) * (nx / 6.0f);
  for (i = 0; i < nx; i++)
  {
    for (j = 0; j < ny; j++)
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

  // Print initial temperatures only for small problems
  if (nx * ny <= PROBLEMAPICCOLO)
    print(surf, nx, ny);

  // Compute-intensive kernel for the heat simulation
  int index;
  float sij, sim1j, sijm1, sip1j, sijp1;

  // Simulate N time steps
  for (n = 0; n < nsteps; n++)
  {
    // Go through the entire 2D surface
    for (i = 0; i < nx; i++)
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
        surf1[index] = sij + A * DT *
                                 ((sim1j - 2.0 * sij + sip1j) / DX2 + (sijm1 - 2.0 * sij + sijp1) / DY2);
      }
    }
    // Swap the surf and surf1 pointers for the next time step
    // (it avoids copying all data from surf1 to surf)
    tmp = surf;
    surf = surf1;
    surf1 = tmp;
  }

  // Print final temperatures only for small problems
  if (nx * ny <= PROBLEMAPICCOLO)
  {
    printf("\n\n");
    print(surf, nx, ny);
  }

  // Free memory
  free(surf);
  free(surf1);

  return 0;
}

// Take in input 2D coordinates of a point in a matrix
// and translate in a 1D offset for the linearized matrix
int getIndex(int i, int j, int width)
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
