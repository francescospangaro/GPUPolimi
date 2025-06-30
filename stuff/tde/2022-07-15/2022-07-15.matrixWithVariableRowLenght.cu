#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MIN_COLS 1000 
#define MAX_COLS 9000
#define MIN_ROWS 2048
#define MAX_ROWS 2048
#define MAX_VAL 10

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

void elaborateMatrix(int* a, int rows, int* colOffsets, int *cols, int* b);
void printM(int* a, int rows, int* colOffsets, int *cols);
__global__ void compute_on_gpu(int* a, int rows, int* colOffsets, int *cols, int* b);
__global__ void compute_on_gpu2(int* a, int rows, int* colOffsets, int *cols, int* b);
__global__ void compute_on_gpu3(int* a, int rows, int* colOffsets, int *cols, int* b);
__global__ void compute_on_gpu3_child(int* a, int cols, int* b, int coeff);
double get_time();


void elaborateMatrix(int* a, int rows, int* colOffsets, int *cols, int* b){
  int i, j, coeff;
  for(i=0; i<rows; i++){
    coeff = i*cols[i];
    for(j=0; j<cols[i]; j++)
      *(b + colOffsets[i] + j) = *(a + colOffsets[i] + j) + coeff;
      // since both indexes depends on both i and j we can parallelize this computation using 2D thread block
  }
}

__global__ void compute_on_gpu(int* a, int rows, int* colOffsets, int *cols, int* b){
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  int i = blockIdx.y*blockDim.y+threadIdx.y;
  if(i<rows && j<cols[i]){
    *(b + colOffsets[i] + j) = *(a + colOffsets[i] + j) + i*cols[i];    
  }
}

__global__ void compute_on_gpu2(int* a, int rows, int* colOffsets, int *cols, int* b){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int coeff, j;
  if(i<rows){
    coeff = i*cols[i];
    for(j=0; j<cols[i]; j++)
      *(b + colOffsets[i] + j) = *(a + colOffsets[i] + j) + coeff;
  }
}

__global__ void compute_on_gpu3(int* a, int rows, int* colOffsets, int *cols, int* b){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int coeff;
  if(i<rows){
    coeff = i*cols[i];
    compute_on_gpu3_child<<<((cols[i]-1)/BLOCKSIZEY+1), BLOCKSIZEY>>>(a + colOffsets[i], cols[i], b + colOffsets[i], coeff);
  }
}

__global__ void compute_on_gpu3_child(int* a, int cols, int* b, int coeff){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i<cols)
    *(b + i) = *(a + i) + coeff; 
}


//display the matrix on the screen
void printM(int* a, int rows, int* colOffsets, int *cols) {
  int i, j;
  for(i=0; i<rows; i++){
    for(j=0; j<cols[i]; j++)
      printf("%3d ", *(a + colOffsets[i]+ j));
    printf("\n");    
  }
}

// function to get the time of day in seconds
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]){
  int *a, *b, rows, *cols, *colOffsets, i, j, numOfElems;

  // declare timing variables
  double cpu_start, cpu_end, gpu_start1, gpu_end1, gpu_start2, gpu_end2, gpu_start3, gpu_end3;

  srand(0);

  rows = rand() % (MAX_ROWS - MIN_ROWS + 1) + MIN_ROWS;
  cols = (int*) malloc(sizeof(int)*rows);
  if(!cols){
    printf("Error on malloc\n");
    return -1;
  }
  for(i=0; i<rows; i++)
    cols[i] = rand() % (MAX_COLS - MIN_COLS + 1) + MIN_COLS;

  colOffsets = (int*) malloc(sizeof(int)*rows);
  if(!colOffsets){
    printf("Error on malloc\n");
    return -1;
  }
  for (i=0, numOfElems=0; i<rows; i++){
    colOffsets[i] = numOfElems;
    numOfElems += cols[i];
  }

  a = (int*) malloc(sizeof(int)*numOfElems);
  if(!a){
    printf("Error on malloc\n");
    return -1;
  }
  for (i=0; i<numOfElems; i++)
    a[i] = rand() % MAX_VAL;

  b = (int*) malloc(sizeof(int)*numOfElems);
  if(!b){
    printf("Error on malloc\n");
    return -1;
  }

  //execute on CPU  
  cpu_start = get_time();
  elaborateMatrix(a, rows, colOffsets, cols, b);
  cpu_end = get_time();
  
//  printM(a, rows, colOffsets, cols);
//  printf("\n");
//  printM(b, rows, colOffsets, cols);

  int *a_d, *b_d, *cols_d, *colOffsets_d;
  int *b_gpu, maxcols;

  b_gpu = (int*) malloc(sizeof(int)*numOfElems);
  if(!b_gpu){
    printf("Error on malloc\n");
    return -1;
  }

  for(i=1, maxcols = cols[0]; i<rows; i++)
    if(maxcols<cols[i])
      maxcols = cols[i]; 

  // allocate device space
  CHECK(cudaMalloc((void**)&a_d, numOfElems * sizeof(int)));
  CHECK(cudaMalloc((void**)&b_d, numOfElems * sizeof(int)));
  CHECK(cudaMalloc((void**)&cols_d, rows * sizeof(int)));
  CHECK(cudaMalloc((void**)&colOffsets_d, rows * sizeof(int)));

  // copy vector data from host to device
  CHECK(cudaMemcpy(a_d, a, numOfElems * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(cols_d, cols, rows * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(colOffsets_d, colOffsets, rows * sizeof(int), cudaMemcpyHostToDevice));

  // set #blocks and #threads
  dim3 blocksPerGrid((maxcols-1)/BLOCKSIZEX+1, (rows-1)/BLOCKSIZEY+1, 1);
  dim3 threadsPerBlock(BLOCKSIZEX, BLOCKSIZEY, 1);

//  printf("-> %d %d\n", blocksPerGrid.x, blocksPerGrid.y);

  // execute the kernel 1
  gpu_start1 = get_time();
  compute_on_gpu<<<blocksPerGrid, threadsPerBlock>>>(a_d, rows, colOffsets_d, cols_d, b_d);
  CHECK_KERNELCALL()
  cudaDeviceSynchronize();
  gpu_end1 = get_time();

  // copy the result back to the device
  CHECK(cudaMemcpy(b_gpu, b_d, numOfElems * sizeof(int), cudaMemcpyDeviceToHost));

//  printf("\n");
//  printM(b_gpu, rows, colOffsets, cols);
//  printf("\n");

  // result check
  int check = 1;
  for(i=0; i<rows; i++){
    for(j=0; j<cols[i]; j++)
      if(*(b + colOffsets[i]+ j) != *(b_gpu + colOffsets[i]+ j)){
        printf("1- Error in b[%d][%d]:  %d != %d \n", i, j, *(b + colOffsets[i]+ j), *(b_gpu + colOffsets[i]+ j));
        check = 0;
      }
  }

  // set #blocks and #threads
  dim3 blocksPerGrid2((rows-1)/BLOCKSIZEY+1, 1, 1);
  dim3 threadsPerBlock2(BLOCKSIZEY, 1, 1);

  // execute the kernel 2
  gpu_start2 = get_time();
  compute_on_gpu2<<<blocksPerGrid2, threadsPerBlock2>>>(a_d, rows, colOffsets_d, cols_d, b_d);
  CHECK_KERNELCALL()
  cudaDeviceSynchronize();
  gpu_end2 = get_time();

  // copy the result back to the device
  CHECK(cudaMemcpy(b_gpu, b_d, numOfElems * sizeof(int), cudaMemcpyDeviceToHost));

//  printf("\n");
//  printM(b_gpu, rows, colOffsets, cols);
//  printf("\n");

  // result check
  for(i=0; i<rows; i++){
    for(j=0; j<cols[i]; j++)
      if(*(b + colOffsets[i]+ j) != *(b_gpu + colOffsets[i]+ j)){
        printf("2- Error in b[%d][%d]:  %d != %d \n", i, j, *(b + colOffsets[i]+ j), *(b_gpu + colOffsets[i]+ j));
        check = 0;
      }
  }

  dim3 blocksPerGrid3((rows-1)/BLOCKSIZEY+1, 1, 1);
  dim3 threadsPerBlock3(BLOCKSIZEY, 1, 1);

  // execute the kernel 3
  gpu_start3 = get_time();
  compute_on_gpu3<<<blocksPerGrid3, threadsPerBlock3>>>(a_d, rows, colOffsets_d, cols_d, b_d);
  //CHECK_KERNELCALL()
  cudaDeviceSynchronize();
  gpu_end3 = get_time();

  // copy the result back to the device
  CHECK(cudaMemcpy(b_gpu, b_d, numOfElems * sizeof(int), cudaMemcpyDeviceToHost));

//  printf("\n");
//  printM(b_gpu, rows, colOffsets, cols);
//  printf("\n");

  // result check
  for(i=0; i<rows; i++){
    for(j=0; j<cols[i]; j++)
      if(*(b + colOffsets[i]+ j) != *(b_gpu + colOffsets[i]+ j)){
        printf("3- Error in b[%d][%d]:  %d != %d \n", i, j, *(b + colOffsets[i]+ j), *(b_gpu + colOffsets[i]+ j));
        check = 0;
      }
  }


  if(check){
    printf("All results correct!\n");
    printf("CPU Time:  %.5lf\nGPU Time1: %.5lf\nGPU Time2: %.5lf\nGPU Time3: %.5lf\n", 
            cpu_end-cpu_start, gpu_end1-gpu_start1, gpu_end2-gpu_start2, gpu_end3-gpu_start3);
  }

  CHECK(cudaFree(a_d));
  CHECK(cudaFree(b_d));
  CHECK(cudaFree(cols_d));
  CHECK(cudaFree(colOffsets_d));

  free(a);
  free(b);
  free(b_gpu);
  free(cols);
  free(colOffsets);

  return 0;
}