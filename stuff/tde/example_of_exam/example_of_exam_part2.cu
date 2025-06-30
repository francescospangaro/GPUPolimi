/*
* The kernel function to accelerate receives in input a vector of positive integers, called A, 
* together with its size, and a second empty vector of integers, B, of the same size. 
* For each element i in A, the function saves in B[i] the value 1 if A[i] is greater than all the 
* neighbor values with an index between (i-DIST) and (i+DIST), bounds included and if they exist; 
* 0 otherwise. DIST is a constant value defined with a macro.
* The main function is a dummy program that receives as an argument the vector size, instantiates and 
* populates randomly A, invokes the above function, and shows results.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


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
__global__ void compute_on_gpu(int *V, int *R, int num);
__global__ void compute_on_gpu2(int *V, int *R, int num);
double get_time();

//display a vector of numbers on the screen
void printV(int *V, int num) {
  int i;
  for(i=0; i<num; i++)
      printf("%3d(%d) ", V[i], i);
  printf("\n");    
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

__global__ void compute_on_gpu(int *V, int *R, int num){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j, ok;
  if (i<num){
    for(j=-DIST, ok=1; j<=DIST; j++){
      if(i+j>=0 && i+j<num && j!=0 && V[i]<=V[i+j])
        ok=0;
    }
    R[i] = ok;
  }
}

__global__ void compute_on_gpu2(int *V, int *R, int num){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ int tile[BLOCKSIZE+2*DIST];
  
  tile[DIST+threadIdx.x] = V[blockIdx.x*blockDim.x+threadIdx.x];
  if(threadIdx.x<DIST){
    if(blockIdx.x>0)
      tile[threadIdx.x] = V[blockIdx.x*blockDim.x+threadIdx.x-DIST];
    else
      tile[threadIdx.x] = 0;
  }
  if(threadIdx.x>=blockDim.x-DIST){
    if(blockIdx.x<gridDim.x-1)
      tile[threadIdx.x+DIST*2] = V[(blockIdx.x)*blockDim.x+threadIdx.x+DIST];
    else
      tile[DIST*2+threadIdx.x] = 0;  
  }
  __syncthreads();

  int j, ok;
  if (i<num){
    for(j=threadIdx.x, ok=1; j<=DIST*2+threadIdx.x; j++){
      if(j!=threadIdx.x+DIST && tile[threadIdx.x+DIST]<=tile[j])
        ok=0;
    }
    R[i] = ok;
  }
}

// function to get the time of day in seconds
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}


int main(int argc, char **argv) {
  int *A;
  int *B;
  int dim;
  int i;

  // declare timing variables
  double cpu_start, cpu_end, gpu_start1, gpu_end1, gpu_start2, gpu_end2;

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
  cpu_start = get_time();
  compute(A, B, dim);
  cpu_end = get_time();

  // declare GPU vars
  int *A_d, *B_d;
  

  // allocate host vector that will contain the GPU-computed results
  int *B_GPU = (int*)malloc(dim * sizeof(int));
  if(!B_GPU){
    printf("Error: malloc failed\n");
    return 1;
  }

  // allocate device space
  CHECK(cudaMalloc((void**)&A_d, dim * sizeof(int)));
  CHECK(cudaMalloc((void**)&B_d, dim * sizeof(int)));

  // copy vector data from host to device
  CHECK(cudaMemcpy(A_d, A, dim * sizeof(int), cudaMemcpyHostToDevice));

  // set #blocks and #threads
  dim3 blocksPerGrid((dim-1)/BLOCKSIZE+1, 1, 1);
  dim3 threadsPerBlock(BLOCKSIZE, 1, 1);

  // execute the kernel 1
  gpu_start1 = get_time();
  compute_on_gpu<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, dim);
  //CHECK_KERNELCALL()
  cudaDeviceSynchronize();
  gpu_end1 = get_time();

  // copy the result back to the device
  CHECK(cudaMemcpy(B_GPU, B_d, dim * sizeof(int), cudaMemcpyDeviceToHost));

  // result check
  int check = 1;
  for(i = 0; i < dim; i++){
    if(B[i] != B_GPU[i]){
      printf("1- Error in B[%d]!  %d %d \n", i, B[i], B_GPU[i]);
      check = 0;
      break;
    }
  }

  // execute the kernel 2
  gpu_start2 = get_time();
  compute_on_gpu2<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, dim);
  //CHECK_KERNELCALL()
  cudaDeviceSynchronize();
  gpu_end2 = get_time();

  // copy the result back to the device
  CHECK(cudaMemcpy(B_GPU, B_d, dim * sizeof(int), cudaMemcpyDeviceToHost));

  // result check
  for(i = 0; i < dim && check; i++){
    if(B[i] != B_GPU[i]){
      printf("2- Error in B[%d]:  %d != %d \n", i, B[i], B_GPU[i]);
      check = 0;
    }
  }

  if(check){
    printf("All results correct!\n");
    printf("CPU Time:  %.5lf\nGPU Time1: %.5lf\nGPU Time2: %.5lf\n", 
            cpu_end-cpu_start, gpu_end1-gpu_start1, gpu_end2-gpu_start2);
  }

      
  //print results
  //printV(A, dim);
  //printV(B, dim);  
  
  free(A);
  free(B);
  free(B_GPU);

  CHECK(cudaFree(A_d));
  CHECK(cudaFree(B_d));
  
  return 0;
}


