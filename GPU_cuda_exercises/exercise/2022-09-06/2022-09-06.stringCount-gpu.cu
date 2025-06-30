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

#include <stdio.h>
#include <stdlib.h>

#define MAXVAL 2
#define BLOCK_DIM 32

void printText(char *text, int num);
void findString(char *text, int textDim, char *str, int strDim, char *match);
int countMatches(char *match, int num);

// display a vector of numbers on the screen
void printText(char *text, int num)
{
  int i;
  for (i = 0; i < num; i++)
    printf("%c", text[i]);
  printf("\n");
}

// kernel function 1: identify strings in the text
void findString(char *text, int textDim, char *str, int strDim, char *match)
{
  int i, j, ok;
  for (i = 0; i < textDim - strDim + 1; i++)
  {
    for (j = 0, ok = 1; j < strDim && ok; j++)
    {
      if (text[i + j] != str[j])
        ok = 0;
    }
    match[i] = ok;
  }
}

__global__ void findString_gpu(char *text, int textDim, char *str, int strDim, char *match)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int j, ok;
  if (tid < textDim - strDim + 1)
  {
    for (j = 0, ok = 1; j < strDim && ok; j++)
    {
      if (text[tid + j] != str[j])
        ok = 0;
    }
    match[tid] = ok;
  }
}

__global__ void findString_shared_gpu(char *text, int textDim, char *str, int strDim, char *match) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ char tile_s[];

  for(int i=threadIdx.x;i<BLOCK_DIM+strDim;i+=BLOCK_DIM) {
    tile_s[i] = text[i + BLOCK_DIM*blockIdx.x];
  }
  __syncthreads();

  int j, ok;
  if (tid < textDim - strDim + 1)
  {
    for (j = 0, ok = 1; j < strDim && ok; j++)
    {
      if (tile_s[threadIdx.x + j] != str[j])
        ok = 0;
    }
    match[tid] = ok;
  }

}

// kernel function 2: count matches
int countMatches(char *match, int num)
{
  int i, count;
  for (i = 0, count = 0; i < num; i++)
    count += match[i];
  return count;
}

__global__ void countMatches_gpu(char *match, int *count, int num)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num)
  {
    atomicAdd(count, match[tid]);
  }
}

int main(int argc, char **argv)
{
  char *text, *str, *match;
  int count;
  int textDim, strDim;
  int i;

  // read arguments
  if (argc != 3)
  {
    printf("Please specify sizes of the two input vectors\n");
    return 0;
  }
  textDim = atoi(argv[1]);
  strDim = atoi(argv[2]);

  // allocate memory for the three vectors
  text = (char *)malloc(sizeof(char) * (textDim));
  if (!text)
  {
    printf("Error: malloc failed\n");
    return 1;
  }
  str = (char *)malloc(sizeof(char) * (strDim));
  if (!str)
  {
    printf("Error: malloc failed\n");
    return 1;
  }
  match = (char *)malloc(sizeof(char) * (textDim - strDim + 1));
  if (!match)
  {
    printf("Error: malloc failed\n");
    return 1;
  }

  // initialize input vectors
  srand(0);
  for (i = 0; i < textDim; i++)
    text[i] = rand() % MAXVAL + 'a';

  for (i = 0; i < strDim; i++)
    str[i] = rand() % MAXVAL + 'a';

  // execute on CPU
  findString(text, textDim, str, strDim, match);
  count = countMatches(match, textDim - strDim + 1);

  // print results
  printText(text, textDim);
  printText(str, strDim);
  printf("CPU res:\n");
  printf("%d\n", count);

  char *d_text, *d_str, *d_match;
  int *d_count;
  cudaMalloc(&d_text, sizeof(char) * textDim);
  cudaMalloc(&d_str, sizeof(char) * strDim);
  cudaMalloc(&d_match, sizeof(char) * (textDim - strDim + 1));
  cudaMalloc(&d_count, sizeof(int));

  cudaMemcpy(d_text, text, sizeof(char) * textDim, cudaMemcpyHostToDevice);
  cudaMemcpy(d_str, str, sizeof(char) * strDim, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BLOCK_DIM);
  dim3 blocksPerGrid((textDim - strDim) / BLOCK_DIM + 1);

  //findString_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_text, textDim, d_str, strDim, d_match);
  findString_shared_gpu<<<blocksPerGrid, threadsPerBlock, sizeof(char)*(BLOCK_DIM+strDim)>>>(d_text,textDim,d_str,strDim,d_match);
  cudaDeviceSynchronize();

  cudaMemset(&d_count, 0, sizeof(int));
  countMatches_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_match, d_count, textDim - strDim + 1);

  int gpu_res;
  cudaMemcpy(&gpu_res, d_count, sizeof(int), cudaMemcpyDeviceToHost);

  printf("GPU res:\n");
  printf("%d\n", gpu_res);
  if (gpu_res != count)
  {
    printf("CPU and GPU results are NOT equivalent...\n");
    exit(EXIT_FAILURE);
  }

  printf("ALL OK...\n");

  cudaDeviceReset();
  free(text);
  free(str);
  free(match);

  return 0;
}
