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

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define BLOCKSIZE 32

#define A 0.5f    // Diffusion constant
#define DX 0.01f  // Horizontal grid spacing 
#define DY 0.01f  // Vertical grid spacing
#define DX2 (DX*DX)
#define DY2 (DY*DY)
#define DT (DX2 * DY2 / (2.0f * A * (DX2 + DY2))) // Largest stable time step
#define PROBLEMAPICCOLO 100
#define EPS 0.001f

void print(float *surf, int nx, int ny);
int check(float *surf1, float *surf2, int nx, int ny);
__host__ __device__ int getIndex(int i, int j, int width);
__global__ void compute_on_gpu(float *surf, float *surf1, int nx, int ny);
__global__ void initialize_on_gpu(float *surf, int nx, int ny);
double get_time();


__global__ void compute_on_gpu(float *surf, float *surf1, int nx, int ny) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  float sij, sim1j, sijm1, sip1j, sijp1;

  if(index < nx*ny) {
    int i = index / ny;
    int j = index % ny;

    // Compute the heat transfer taking old temperature from surf
    // and saving new temperature in surf1
    index = getIndex(i, j, ny);
    sij = surf[index];
    if(i>0)
      sim1j = surf[getIndex(i-1, j, ny)];
    else
      sim1j = sij;
    if(j>0)
      sijm1 = surf[getIndex(i, j-1, ny)];
    else
      sijm1 = sij;
    if(i<nx-1)
      sip1j = surf[getIndex(i+1, j, ny)];
    else
      sip1j = sij;
    if(j<ny-1)
      sijp1 = surf[getIndex(i, j+1, ny)];
    else
      sijp1 = sij;
    surf1[index] = sij + A * DT * 
        ( (sim1j - 2.0f*sij + sip1j)/DX2 + (sijm1 - 2.0f*sij + sijp1)/DY2 );
  }  
}

__global__ void initialize_on_gpu(float *surf, int nx, int ny) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index < nx*ny){
    int i = index / ny;
    int j = index % ny;
    float radius2 = (nx/6.0f) * (nx/6.0f);
    // Distance of point i, j from the origin
    float ds2 = (i - nx/2) * (i - nx/2) + (j - ny/2)*(j - ny/2);
    if (ds2 < radius2)
      surf[index] = 65.0f;
    else
      surf[index] = 5.0f;
  }
}

int main(int argc, char **argv) {
  float *surf, *surf1, *tmp;
  float *surf_for_gpu, *surf_gpu, *surf1_gpu, *tmp_gpu;
  int i, j, nx, ny, nsteps, n;

  // declare timing variables
  double cpu_start, cpu_end, gpu_start1, gpu_end1;

  // Read arguments
  if(argc != 4){
    printf("Please specify sizes of the 2D surface and the number of time steps\n");
    return 0;
  }
  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  nsteps = atoi(argv[3]);
  
  // Allocate memory for the two arrays
  surf = (float*) malloc(sizeof(float) * nx * ny);
  if(!surf){
    printf("Error: malloc failed\n");
    return 1;
  }
  surf1 = (float*) malloc(sizeof(float) * nx * ny);
  if(!surf1){
    printf("Error: malloc failed\n");
    return 1;
  }

  // Initialize the data with a pattern of disk of radius of 1/6 of the width
  float radius2 = (nx/6.0f) * (nx/6.0f);
  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      int index = getIndex(i, j, ny);
      // Distance of point i, j from the origin
      float ds2 = (i - nx/2) * (i - nx/2) + (j - ny/2)*(j - ny/2);
      if (ds2 < radius2)
        surf[index] = 65.0f;
      else
        surf[index] = 5.0f;
    }
  }

  // Print initial temperatures only for small problems
  if(nx*ny <= PROBLEMAPICCOLO)
    print(surf, nx, ny); 
  
  // Compute-intensive kernel for the heat simulation onto CPU
  int index;
  float sij, sim1j, sijm1, sip1j, sijp1;

  // Simulate N time steps
  cpu_start = get_time();
  for (n = 0; n < nsteps; n++) {
    // Go through the entire 2D surface
    for (i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        // Compute the heat transfer taking old temperature from surf
        // and saving new temperature in surf1
        index = getIndex(i, j, ny);
        sij = surf[index];
        if(i>0)
          sim1j = surf[getIndex(i-1, j, ny)];
        else
          sim1j = sij;

        if(j>0)
          sijm1 = surf[getIndex(i, j-1, ny)];
        else
          sijm1 = sij;

        if(i<nx-1)
          sip1j = surf[getIndex(i+1, j, ny)];
        else
          sip1j = sij;

        if(j<ny-1)
          sijp1 = surf[getIndex(i, j+1, ny)];
        else
          sijp1 = sij;

        surf1[index] = sij + A * DT * 
            ( (sim1j - 2.0f*sij + sip1j)/DX2 + (sijm1 - 2.0f*sij + sijp1)/DY2 );
      }
    }
    // Swap the surf and surf1 pointers for the next time step
    // (it avoids copying all data from surf1 to surf) 
    tmp = surf;
    surf = surf1;
    surf1 = tmp;
  }
  cpu_end = get_time();
  
  // Print final temperatures only for small problems
  if(nx*ny <= PROBLEMAPICCOLO){
    printf("\noutput on CPU:\n");
    print(surf, nx, ny); 
  }
  
  // Allocate memory for the GPU computation
  surf_for_gpu = (float*) malloc(sizeof(float) * nx * ny);
  if(!surf_for_gpu){
    printf("Error: malloc failed\n");
    return 1;
  }

  // Allocate GPU memory for kernel 1
  CHECK(cudaMalloc((void**)&surf_gpu, nx * ny * sizeof(float)));
  CHECK(cudaMalloc((void**)&surf1_gpu, nx * ny * sizeof(float)));

  // Initialize input data on GPU
  //CHECK(cudaMemcpy(surf_gpu, surf_for_gpu, nx * ny * sizeof(float), cudaMemcpyHostToDevice));
  // set #blocks and #threads for the two kernels
  dim3 blocksPerGrid((nx*ny-1)/BLOCKSIZE+1, 1, 1);
  dim3 threadsPerBlock(BLOCKSIZE, 1, 1);
  initialize_on_gpu<<<blocksPerGrid, threadsPerBlock>>>(surf_gpu, nx, ny);
  //CHECK_KERNELCALL()

  // execute the compute kernel
  gpu_start1 = get_time();
  for (n = 0; n < nsteps; n++){
    compute_on_gpu<<<blocksPerGrid, threadsPerBlock>>>(surf_gpu, surf1_gpu, nx, ny);
    //CHECK_KERNELCALL()
    tmp_gpu = surf1_gpu;
    surf1_gpu = surf_gpu;
    surf_gpu = tmp_gpu; 
  }
  cudaDeviceSynchronize();
  gpu_end1 = get_time();

  // copy vector data from host to device for kernel 1
  CHECK(cudaMemcpy(surf_for_gpu, surf_gpu, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));

  // Print final temperatures only for small problems
  if(nx*ny <= PROBLEMAPICCOLO){
    printf("\nOutput on GPU:\n");
    print(surf_for_gpu, nx, ny); 
  }

  printf("\nGPU: ");
  if(check(surf_for_gpu, surf, nx, ny))
    printf("OK!\n\n");
  else
    printf("Error!\n\n");

  printf("CPU Time:  %.5lf\nGPU Time1: %.5lf\n", 
          cpu_end-cpu_start, gpu_end1-gpu_start1);


  // Free memory
  free(surf);
  free(surf1);
  free(surf_for_gpu);

  CHECK(cudaFree(surf_gpu));
  CHECK(cudaFree(surf1_gpu));
  
  return 0;
}

// Take in input 2D coordinates of a point in a matrix
// and translate in a 1D offset for the linearized matrix
__host__ __device__ int getIndex(int i, int j, int width) {
  return (i*width + j);
}

// Display a vector of numbers on the screen
void print(float *surf, int nx, int ny){
  int i, j;
  for(i=0; i<nx; i++){
    for(j=0; j<ny; j++)
      printf("%6.3f ", surf[getIndex(i, j, ny)]);
    printf("\n");    
  }
}

// Check if two arrays are equal
int check(float *surf1, float *surf2, int nx, int ny) {
  int i;
  for(i=0; i<ny*nx; i++)
    if (fabs(surf1[i] - surf2[i])>EPS)
      return 0;
  return 1;
}

// Get the time of day in seconds
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}