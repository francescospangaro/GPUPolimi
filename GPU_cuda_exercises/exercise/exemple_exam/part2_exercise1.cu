#include <stdio.h>
#include <stdlib.h>

#define MAXVAL 100
#define DIST 10
#define BLOCK_DIM 1024

void compute(int *V, int *R, int num);

// kernel function: identify peaks in the vector
void compute(int *V, int *R, int num)
{
  int i, j, ok;
  for (i = 0; i < num; i++)
  {
    for (j = -DIST, ok = 1; j <= DIST; j++)
    {
      if (i + j >= 0 && i + j < num && j != 0 && V[i] <= V[i + j])
        ok = 0;
    }
    R[i] = ok;
  }
}

__global__ void compute_gpu(int *V, int *R, int num)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < num)
  {
    for (int j = -DIST; j <= DIST; j++)
    {
      int ok = 1;
      if (tid + j >= 0 && tid + j < num && j != 0 && V[tid] <= V[tid + j])
      {
        ok = 0;
      }
      R[tid] = ok;
    }
  }
}

int main(int argc, char **argv) { 
  int *A; 
  int *B; 
  int dim; 
  int i; 
 
  //read arguments 
  if(argc!=2){ 
    printf("Please specify sizes of the input vector\n"); 
    return 0; 
  } 
  dim=atoi(argv[1]); 
   
  //allocate memory for the three vectors 
  A = (int*) malloc(sizeof(int) * dim); 
  B = (int*) malloc(sizeof(int) * dim); 
 
  //initialize input vectors 
  /*code omitted for the sake of space*/ 
 
  //execute on CPU   
  compute(A, B, dim); 
  
  //print results 
  /*code omitted for the sake of space*/ 
 
  free(A); 
  free(B); 
   
  return 0; 
}