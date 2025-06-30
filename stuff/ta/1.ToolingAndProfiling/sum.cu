#include<stdio.h>
#include<cuda_runtime.h>

#define N 32
#define BLOCKDIM N

#define CHECK(call) \
{ \
  const cudaError_t err = call; \
  if (err != cudaSuccess) { \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} \


#define CHECK_KERNELCALL() \
{ \
  const cudaError_t err = cudaGetLastError();\
  if (err != cudaSuccess) {\
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
  }\
}\

// A is constant it can be placed in texture memory
// Thus enables the usage of read-only cache
__global__ void vsumKernel(const char* __restrict__ a, char* b){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO insert code here
}

int main(){
  char h_va[N], h_vb[N];
  char *d_va, *d_vb;
  int i;

  /*initialize vectors*/  
  for(i=0; i<N; i++){
    h_va[i] = i;
  }
  
  /*allocate memory on the GPU*/
  CHECK(cudaMalloc(&d_va, N*sizeof(char)));
  CHECK(cudaMalloc(&d_vb, N*sizeof(char)));

  /*transmit data to GPU*/
  CHECK(cudaMemcpy(d_va, h_va, N*sizeof(char), cudaMemcpyHostToDevice));

  /*invoke the kernel on the GPU*/
  dim3 blocksPerGrid((N+BLOCKDIM-1)/BLOCKDIM, 1, 1);
  dim3 threadsPerBlock(BLOCKDIM, 1, 1);
  printf("%d %d\n", blocksPerGrid.x, threadsPerBlock.x);
  vsumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_va, d_vb);
  CHECK_KERNELCALL();
  
  /*transmit data from the GPU*/
  CHECK(cudaMemcpy(h_vb, d_vb, N*sizeof(char), cudaMemcpyDeviceToHost));

  /*free memory*/
  CHECK(cudaFree(d_va));
  CHECK(cudaFree(d_vb));
  
  return 0;
}



