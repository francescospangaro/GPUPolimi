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

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 32
#define MAXVAL 2

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
int countMatches(char *match, int num);
__global__ void findStringKernel(char* text, int textDim, char* str, int strDim, char* match);
__global__ void countMatchesKernel(char *match, int num, int res);
__global__ void findStringKernelShared(char* text, int textDim, char* str, int strDim, char* match);

__global__ void findStringKernel(char* text, int textDim, char* str, int strDim, char* match) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j, ok;
  if (i < (textDim-strDim+1)) {
    for(j=0, ok=1; j<strDim && ok; j++){
      if(text[i+j]!=str[j])
        ok=0;
    }
    match[i] = ok;
  }
}

__global__ void findStringKernelShared(char* text, int textDim, char* str, int strDim, char* match) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  extern __shared__ char tile[];
  int j, ok;

  for (j = threadIdx.x; j < BLOCKSIZE+strDim-1; j+=BLOCKSIZE)
    tile[j] = text[blockIdx.x * blockDim.x + j];
  __syncthreads();

  if (i < (textDim-strDim+1)) {
    for(j=0, ok=1; j<strDim && ok; j++){
      if(tile[i+j]!=str[j])
        ok=0;
    }
    match[i] = ok;
  }
}

__global__ void countMatchesKernel(char *match, int num, int *res) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num){
    atomicAdd(res, match[i]);
  }
}

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

//kernel function 2: count matches
int countMatches(char *match, int num) {
  int i, count;
  for(i=0, count=0; i<num; i++)
    count+=match[i];
  return count;    
}

int main(int argc, char **argv) {
  char *text, *str, *match;
  int count;
  int textDim, strDim;
  int i;

  //read arguments
  if(argc!=3){
    printf("Please specify sizes of the two input vectors\n");
    return 0;
  }
  textDim=atoi(argv[1]);
  strDim=atoi(argv[2]);
  
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
  match = (char*) malloc(sizeof(char) * (textDim-strDim+1));
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
  
  char *text_d, *str_d, *match_d;
  int *res;
  CHECK(cudaMalloc((void**) &res, sizeof(int)));
  CHECK(cudaMalloc((void**) &text_d, sizeof(char) * textDim));
  CHECK(cudaMalloc((void**) &str_d, sizeof(char) * strDim));
  CHECK(cudaMalloc((void**) &match_d, sizeof(char) * (textDim-strDim+1)));

  int *(count_d);
  *count_d = 0;
  CHECK(cudaMemcpy(res, count_d, sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(text_d, text, sizeof(char) * textDim, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(str_d, str, sizeof(char) * strDim, cudaMemcpyHostToDevice));

  dim3 blocksPerGrid((textDim - strDim) / BLOCKSIZE +1);
  dim3 threadsPerBlock(BLOCKSIZE);

  findStringKernel<<<blocksPerGrid, threadsPerBlock>>>(text_d, textDim, str_d, strDim, match_d);
  CHECK_KERNELCALL();
  countMatchesKernel<<<blocksPerGrid, threadsPerBlock>>>(match_d, textDim-strDim+1, res);
  CHECK_KERNELCALL();
  CHECK(cudaMemcpy(count_d, res, sizeof(int), cudaMemcpyDeviceToHost));

  findStringKernelShared<<<blocksPerGrid, threadsPerBlock, (BLOCKSIZE + strDim) * sizeof(char)>>>(text_d, textDim, str_d, strDim, match_d);
  CHECK_KERNELCALL();
  countMatchesKernel<<<blocksPerGrid, threadsPerBlock>>>(match_d, textDim-strDim+1, res);
  CHECK_KERNELCALL();
  CHECK(cudaMemcpy(count_d, res, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(res));
  CHECK(cudaFree(text_d));
  CHECK(cudaFree(str_d));
  CHECK(cudaFree(match_d));
  
  //execute on CPU  
  findString(text, textDim, str, strDim, match);
  count = countMatches(match, textDim-strDim+1);
 
  //print results
  printText(text, textDim);
  printText(str, strDim);
  printf("%d\n", count);
  
  free(text);
  free(str);
  free(match);
  
  return 0;
}
