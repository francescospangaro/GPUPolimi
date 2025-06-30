/*
 * The kernel function 1 (mult) performs the multiplication of a vector by a scalar value.
 * The kernel function 2 (compare) receives two vectors of integers, called A and B,
 * together with the sizes sa and sb, and a third empty vector of integers, C, which 
 * size is sa*sb. 
 * For each pair A[i] and B[j], the function saves in C[i][j] value 1 if A[i] > B[j], 
 * 0 otherwise (do consider that the function manages C as a linearized array). 
 * The main function is a dummy program receiving in input sa and sb, populating randomly A 
 * and B, invoking the above two functions and showing results.
 */

#include <stdio.h>
#include <stdlib.h>

#define MAXVAL 100
#define VALUE 10
#define BLOCKSIZEX 32
#define BLOCKSIZEY 32

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

void printM(int *M, int numMRows, int numMColumns);
void compare(int *M, int *N, int dm, int dn, int *P);
void mult(int *V, int dim, int fatt, int *P);
__global__ void compare_gpu(int *M, int *N, int dm, int dn, int *P);
__global__ void compare_gpu_shared(int *M, int *N, int dm, int dn, int *P);
__global__ void mult_gpu(int *V, int dim, int fatt, int *P);

//display a matrix on the screen
void printM(int *M, int numMRows, int numMColumns) {
  int i, j;
  for(i=0; i<numMRows; i++){
    for(j=0; j<numMColumns; j++)
      printf("%3d ", M[i * numMColumns + j]);
    printf("\n");
  }
  printf("\n");
}

//kernel function 1: vector per scalar multiplication
void mult(int *V, int dim, int fatt, int *P){
  int i;
  for(i=0; i<dim; i++)
    P[i] = V[i] * fatt;
}

//kernel function 2: compare each element of M against any element of N
void compare(int *M, int *N, int dm, int dn, int *P){
  int i, j;
  for(i=0; i<dm; i++)
    for(j=0; j<dn; j++)
      P[i * dn + j] = (M[i] > N[j]);
}

//kernel function 1 offloaded onto GPU
__global__ void mult_gpu(int *V, int dim, int fatt, int *P) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<dim) {
    P[i] = V[i] * fatt;
  }  
}

//kernel function 2 offloaded onto GPU
__global__ void compare_gpu(int *M, int *N, int dm, int dn, int *P){
  int index_x = blockIdx.x*blockDim.x+threadIdx.x;
  int index_y = blockIdx.y*blockDim.y+threadIdx.y;
  if (index_y<dm && index_x<dn) {
    P[index_y * dn + index_x] = M[index_y] > N[index_x];
  }    
}

//kernel function 2 offloaded onto GPU and exploiting shared memory
__global__ void compare_gpu_shared(int *M, int *N, int dm, int dn, int *P){
  int index_x = blockIdx.x*blockDim.x+threadIdx.x;
  int index_y = blockIdx.y*blockDim.y+threadIdx.y;
  __shared__ int M_tile[BLOCKSIZEY];
  __shared__ int N_tile[BLOCKSIZEX];
  
    //threads in the first row and the first column of the block read input data
    if(threadIdx.y == 0 && index_x<dn)
      N_tile[threadIdx.x] = N[index_x];

    if(threadIdx.x == 0 && index_y<dm)
      M_tile[threadIdx.y] = M[index_y];
    __syncthreads();

  if (index_y<dm && index_x<dn) {
    P[index_y * dn + index_x] = M_tile[threadIdx.y] > N_tile[threadIdx.x];
  }    
}


int main(int argc, char **argv) {
  int *A, *B, *A1, *B1, *C;
  int sa, sb;
  int i;

  //read arguments
  if(argc!=3){
    printf("Please specify sizes of vectors A and B\n");
    return 0;
  }
  sa=atoi(argv[1]);
  sb=atoi(argv[2]);
  
  //allocate memory for the three vectors
  A = (int*) malloc(sizeof(int) * sa);
  if(!A){
    printf("Error: malloc failed\n");
    return 1;
  }
  A1 = (int*) malloc(sizeof(int) * sa);
  if(!A1){
    free(A);
    printf("Error: malloc failed\n");
    return 1;
  }
  B = (int*) malloc(sizeof(int) * sb);
  if(!B){
    printf("Error: malloc failed\n");
    free(A);
    return 1;
  }
  B1 = (int*) malloc(sizeof(int) * sb);
  if(!B1){
    printf("Error: malloc failed\n");
    free(A);
    free(A1);
    free(B);
    return 1;
  }
  C = (int*) malloc(sizeof(int) * sa*sb);
  if(!C){
    printf("Error: malloc failed\n");
    free(A);
    free(A1);
    free(B);
    free(B1);
    return 1;
  }
  //initialize input vectors A and B
  srand(0);
  for(i=0; i<sa; i++)
    A[i] = rand()%MAXVAL;
  for(i=0; i<sb; i++)
    B[i] = rand()%MAXVAL;

  // execute on CPU
  mult(A, sa, VALUE, A1);
  mult(B, sb, VALUE, B1);
  compare(A1, B1, sa, sb, C);
      
  //print results
  //printM(A, 1, sa);
  //printM(B, 1, sb);
  //printM(C, sa, sb);

  // declare GPU vars
  int *A_d, *B_d, *C_d, *A1_d, *B1_d;

  // allocate host vector that will contain the GPU-computed results
  int *C_GPU = (int*)malloc(sa * sb * sizeof(int));
  if(!C_GPU)
    printf("ERRORE!\n");

  // allocate device space
  CHECK(cudaMalloc((void**)&A_d, sa * sizeof(int)));
  CHECK(cudaMalloc((void**)&B_d, sb * sizeof(int)));
  CHECK(cudaMalloc((void**)&A1_d, sa * sizeof(int)));
  CHECK(cudaMalloc((void**)&B1_d, sb * sizeof(int)));
  CHECK(cudaMalloc((void**)&C_d, sa * sb * sizeof(int)));

  // copy vector data from host to device
  CHECK(cudaMemcpy(A_d, A, sa * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_d, B, sb * sizeof(int), cudaMemcpyHostToDevice));

  // set #blocks and #threads
  dim3 blocksPerGrid1a((sa-1)/BLOCKSIZEX+1, 1, 1);
  dim3 threadsPerBlock1a(BLOCKSIZEX, 1, 1);
  dim3 blocksPerGrid1b((sb-1)/BLOCKSIZEX+1, 1, 1);
  dim3 threadsPerBlock1b(BLOCKSIZEX, 1, 1);
  dim3 blocksPerGrid2((sb-1)/BLOCKSIZEX+1, (sa-1)/BLOCKSIZEY+1, 1);
  dim3 threadsPerBlock2(BLOCKSIZEX, BLOCKSIZEY, 1);

  // execute the kernels
  mult_gpu<<<blocksPerGrid1a, threadsPerBlock1a>>>(A_d, sa, VALUE, A1_d);
  mult_gpu<<<blocksPerGrid1b, threadsPerBlock1b>>>(B_d, sb, VALUE, B1_d);
  compare_gpu<<<blocksPerGrid2, threadsPerBlock2>>>(A1_d, B1_d, sa, sb, C_d);
  CHECK(cudaMemcpy(C_GPU, C_d, sa * sb * sizeof(int), cudaMemcpyDeviceToHost));

  // result check
  int check = 1;
  for(i = 0; i < sa * sb; i++) {
    if(C[i]!=C_GPU[i]){
      printf("Error in C[%d][%d], %d!  %d %d \n", i/sb, i%sb, i, C[i], C_GPU[i]);
      check = 0;
        break;
    }
  }

  // execute the kernel 2 with shared memory
  compare_gpu_shared<<<blocksPerGrid2, threadsPerBlock2>>>(A1_d, B1_d, sa, sb, C_d);
  CHECK(cudaMemcpy(C_GPU, C_d, sa * sb * sizeof(int), cudaMemcpyDeviceToHost));

  // result check
  for(i = 0; i < sa * sb; i++) {
    if(C[i]!=C_GPU[i]){
      printf("Error in C[%d][%d], %d!  %d %d \n", i/sb, i%sb, i, C[i], C_GPU[i]);
      check = 0;
        break;
    }
  }

  if(check){
    printf("ok!\n");
  }  

  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  CHECK(cudaFree(A1_d));
  CHECK(cudaFree(B1_d));
  CHECK(cudaFree(C_d));

  free(A);
  free(B);
  free(A1);
  free(B1);
  free(C);
  
  return 0;
}
