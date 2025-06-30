/*
* This program takes in input a text and a string, being two arrays of char values, and 
* computes how many times the string appears in the text. In particular:
* - function findString receives in input the text and the string and saves in each 
*   position i of a third vector called match, 1 if an occurrence of the string has been 
*   found in the text starting the index i, 0 otherwise.
* - function countMatches receives the vector match in input and count the number of values 
*   equal to 1 (i.e., it counts the number of occurrences of the string in the text).
* - the main function receives as arguments the size of text and string and (for the sake 
*   of brevity) generates randomly the content of the two vectors, invokes the two 
*   functions above and prints the result on the screen.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAXVAL 2
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

void printText(char *text, int num);
void findString(char* text, int textDim, char* str, int strDim, char* match);
void findString2(char* text, int textDim, char* str, int strDim, char* match);
int countMatches(char *match, int num);
__global__ void findString_on_gpu(char* text, int textDim, char* str, int strDim, char* match);
__global__ void countMatches_on_gpu(char *match, int dim, int *ris);
double get_time();


//display a vector of numbers on the screen
void printText(char *text, int num) {
  int i;
  for(i=0; i<num; i++)
    printf("%c", text[i]);
  printf("\n");    
}

//kernel function 1: identify strings in the text
void findString(char* text, int textDim, char* str, int strDim, char* match) {
  int i, j, ok;
  for(i=0; i<textDim-strDim+1; i++){
    for(j=0, ok=1; j<strDim && ok; j++){
      if(text[i+j]!=str[j])
        ok=0;
    }
    match[i] = ok;
  }
}

__global__ void findString_on_gpu(char* text, int textDim, char* str, int strDim, char* match){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j, ok;
  if (i<textDim-strDim+1){
    for(j=0, ok=1; j<strDim && ok; j++){
      if(text[i+j]!=str[j])
        ok=0;
    }
    match[i] = ok; // because of this assignation we can't parallelize using the threadIdx.y otherwise every y thread will write on
                   // the same variable match[i]
  }
}

__global__ void findString_on_gpu2(char* text, int textDim, char* str, int strDim, char* match){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j, ok;
  extern __shared__ char textTile[];

  for(j=threadIdx.x; j<BLOCKSIZE+strDim-1; j+=BLOCKSIZE)
    textTile[j] = text[blockIdx.x*blockDim.x+j];

  __syncthreads();

  if (i<textDim-strDim+1){
    for(j=0, ok=1; j<strDim && ok; j++){
      if(textTile[threadIdx.x+j]!=str[j])
        ok=0;
    }
    match[i] = ok;
  }
}

//kernel function 2: count matches
int countMatches(char *match, int num) {
  int i, count;
  for(i=0, count=0; i<num; i++)
    count+=match[i];
  return count;    
}

__global__ void countMatches_on_gpu(char *match, int dim, int *ris){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<dim){
    atomicAdd(ris, match[i]);
  }
}

// function to get the time of day in seconds
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
  char *text, *str, *match;
  int count;
  int textDim, strDim, matchDim;
  int i;

  // declare timing variables
  double cpu_start, cpu_end, gpu_start1, gpu_end1, gpu_start2, gpu_end2;

  //read arguments
  if(argc!=3){
    printf("Please specify sizes of the two input vectors\n");
    return 0;
  }
  textDim=atoi(argv[1]);
  strDim=atoi(argv[2]);
  matchDim = textDim-strDim+1;
  
  //allocate memory for the three vectors
  text = (char*) malloc(sizeof(char) * (textDim));
  if(!text){
    printf("Error: malloc failed\n");
    return 1;
  }
  str = (char*) malloc(sizeof(char) * (strDim));
  if(!str){
    printf("Error: malloc failed\n");
    return 1;
  }
  match = (char*) malloc(sizeof(char) * matchDim);
  if(!match){
    printf("Error: malloc failed\n");
    return 1;
  }

  //initialize input vectors
  srand(0);
  for(i=0; i<textDim; i++)
    text[i] = rand()%MAXVAL +'a';

  for(i=0; i<strDim; i++)
    str[i] = rand()%MAXVAL +'a';
  
  //execute on CPU  
  cpu_start = get_time();
  findString(text, textDim, str, strDim, match);
  cpu_end = get_time();
  count = countMatches(match, matchDim);
 
  //print results
  printf("%d\n", count);

  char *text_d, *str_d, *match_d;
  int *count_d;

  // allocate host vector that will contain the GPU-computed results
  char *match_gpu = (char*) malloc(sizeof(char) * matchDim);
  if(!match_gpu){
    printf("Error: malloc failed\n");
    return 1;
  }  

  // allocate device space
  CHECK(cudaMalloc((void**)&text_d, textDim * sizeof(char)));
  CHECK(cudaMalloc((void**)&str_d, strDim * sizeof(char)));
  CHECK(cudaMalloc((void**)&match_d, matchDim * sizeof(char)));
  CHECK(cudaMalloc((void**)&count_d, sizeof(int)));

  // copy vector data from host to device
  CHECK(cudaMemcpy(text_d, text, textDim * sizeof(char), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(str_d, str, strDim * sizeof(char), cudaMemcpyHostToDevice));

  // set #blocks and #threads
  dim3 blocksPerGrid((matchDim-1)/BLOCKSIZE+1, 1, 1);
  dim3 threadsPerBlock(BLOCKSIZE, 1, 1);

  // execute the kernel 1
  gpu_start1 = get_time();
  findString_on_gpu<<<blocksPerGrid, threadsPerBlock>>>(text_d, textDim, str_d, strDim, match_d);
  //CHECK_KERNELCALL()
  cudaDeviceSynchronize();
  gpu_end1 = get_time();
  
  cudaMemset(count_d, 0, sizeof(int));
  countMatches_on_gpu<<<blocksPerGrid, threadsPerBlock>>>(match_d, matchDim, count_d);

  // copy the result back to the device
  CHECK(cudaMemcpy(&count, count_d, sizeof(int), cudaMemcpyDeviceToHost));
 
  //print results
  printf("%d\n", count);


  // execute the kernel 2
  gpu_start2 = get_time();
  findString_on_gpu2<<<blocksPerGrid, threadsPerBlock, sizeof(char)*(BLOCKSIZE+strDim-1)>>>(text_d, textDim, str_d, strDim, match_d);
  //CHECK_KERNELCALL()
  cudaDeviceSynchronize();
  gpu_end2 = get_time();
  
  cudaMemset(count_d, 0, sizeof(int));
  countMatches_on_gpu<<<blocksPerGrid, threadsPerBlock>>>(match_d, matchDim, count_d);

  // copy the result back to the device
  CHECK(cudaMemcpy(&count, count_d, sizeof(int), cudaMemcpyDeviceToHost));
 
  //print results
  printf("%d\n", count);

  printf("CPU Time:  %.5lf\nGPU Time1: %.5lf\nGPU Time2: %.5lf\n", 
          cpu_end-cpu_start, gpu_end1-gpu_start1, gpu_end2-gpu_start2);
  
  free(text);
  free(str);
  free(match);
  free(match_gpu);

  CHECK(cudaFree(text_d));
  CHECK(cudaFree(str_d));
  CHECK(cudaFree(match_d));

  
  return 0;
}
