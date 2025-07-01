/*
* The kernel function to accelerate receives in input a vector of positive integers, called A, 
* together with its size, and a second empty vector of integers, B, of the same size. 
* For each element i in A, the function saves in B[i] the value 1 if A[i] is greater than all the 
* neighbor values with an index between (i-DIST) and (i+DIST), bounds included and if they exist; 
* 0 otherwise. DIST is a constant value defined with a macro.
* The main function is a dummy program that receives as an argument the vector size, instantiates and 
* populates randomly A, invokes the above function, and shows results.
*/

#include <stdio.h>
#include <stdlib.h>

#define MAXVAL 100
#define DIST 10
#define BLOCKSIZE 32

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

void printV(int *V, int num);
void compute(int *V, int *R, int num);

//display a vector of numbers on the screen
void printV(int *V, int num) {
  int i;
  for(i=0; i<num; i++)
      printf("%3d(%d) ", V[i], i);
  printf("\n");    
}

__global__ void compute_on_gpu(int *V, int *R, int num);
__global__ void compute_on_gpu2(int *V, int *R, int num);
void compute(int *V, int *R, int num);

__global__ void compute_on_gpu(int *V, int *R, int num) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j, ok;
    if (i < num) {
        for (j = -DIST, ok = 1; i <= DIST; j++) {
            if (i + j >= 0 && i + j < num && j != 0 && V[i] <= V[i+j])
            ok = 0;
        }
        R[i] = ok;
    }
}

__global__ void compute_on_gpu2(int *V, int *R, int num){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int tile[BLOCKSIZE + 2 * DIST];
    int used = threadIdx.x + DIST * 2;

    tile[DIST + threadIdx.x] = V[i];
    if (threadIdx.x < DIST)
        tile[threadIdx.x] = blockIdx.x > 0 ? V[i - DIST] : 0;
    if (threadIdx.x >= blockDim .x - DIST)
        tile[used] = (blockIdx.x < (gridDim.x -1)) ? V[i + DIST] : 0;
    __syncthreads();

    int j, ok;
    if (i < num) {
        for (j = threadIdx.x, ok = 1; j < used; ++j) {
            if (j != threadIdx.x + DIST && tile[threadIdx.x + DIST] <= tile[j])
                ok = 0;
        }
        R[i] = ok;
    }
}

//kernel function: identify peaks in the vector
void compute(int *V, int *R, int num) {
  int i, j, ok;
  for(i=0; i<num; i++){
    for(j=-DIST, ok=1; j<=DIST; j++){
      if(i+j>=0 && i+j<num && j!=0 && V[i]<=V[i+j])
        ok=0;
    }
    R[i] = ok;
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
  if(!A){
    printf("Error: malloc failed\n");
    return 1;
  }
  B = (int*) malloc(sizeof(int) * dim);
  if(!B){
    printf("Error: malloc failed\n");
    return 1;
  }

  //initialize input vectors
  srand(0);
  for(i=0; i<dim; i++)
    A[i] = rand()%MAXVAL +1;
  
  //execute on CPU  
  compute(A, B, dim);
 
  int *A_d;
  int *B_d;
  int *(B_GPU) = (int *) malloc(dim * sizeof(int));

  CHECK(cudaMalloc((void**)&A_d, dim * sizeof(int)));
  CHECK(cudaMalloc((void**)&B_d, dim * sizeof(int)));
  
  CHECK(cudaMemcpy(A_d, A, dim*sizeof(int), cudaMemcpyHostToDevice));

  dim3 blocksPerGrid((dim - 1) / BLOCKSIZE + 1, 1, 1);
  dim3 threadsPerBlock(BLOCKSIZE, 1, 1);

  compute_on_gpu<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, dim);
  cudaDeviceSynchronize();

  CHECK(cudaMemcpy(B_GPU, B_d, dim*sizeof(int), cudaMemcpyDeviceToHost));

  compute_on_gpu2<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, dim);
  cudaDeviceSynchronize();

  CHECK(cudaMemcpy(B_GPU, B_d, dim*sizeof(int), cudaMemcpyDeviceToHost));

  //print results
  printV(A, dim);
  printV(B, dim);
  
  free(A);
  free(B);
  free(B_GPU);
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  
  return 0;
}


