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

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess)                                                         \
    {                                                                               \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                         \
    {                                                                               \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define MAXVAL 2

#define BLOCKDIM 32
#define STRIDE_FACTOR 2

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

// kernel function 2: count matches
int countMatches(char *match, int num)
{
  int i, count;
  for (i = 0, count = 0; i < num; i++)
    count += match[i];
  return count;
}

__global__ void findStringKernel(char const *const __restrict__ text,
                                 int const textDim,
                                 char const *const __restrict__ str,
                                 int const strDim,
                                 char *const __restrict__ match)
{
  int const i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < textDim - strDim + 1)
  {
    int ok = 1;
    for (int j = 0; j < strDim && ok; ++j)
    {
      if (text[i + j] != str[j])
        ok = 0;
    }
    match[i] = ok;
  }
}

__global__ void findStringKernelSharedMem(char const *const __restrict__ text,
                                          int const textDim,
                                          char const *const __restrict__ str,
                                          int const strDim,
                                          char *const __restrict__ match)
{
  int const i = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ char text_tile[];
  if (i < textDim)
    text_tile[threadIdx.x] = text[i];
  if (threadIdx.x >= blockIdx.x + strDim && i + strDim < textDim)
    text_tile[threadIdx.x + strDim] = text[i + strDim];
  __syncthreads();

  if (i < textDim - strDim + 1)
  {
    int ok = 1;
    for (int j = 0; j < strDim && ok; ++j)
    {
      if (text[i + j] != str[j])
        ok = 0;
    }
    match[i] = ok;
  }
}

// kernel function 2: count matches
__global__ void countMatchesKernel(char *__restrict__ match,
                                   int const num,
                                   int *const __restrict__ count)
{
  const int i = STRIDE_FACTOR * threadIdx.x;
  const int offset = blockDim.x * blockIdx.x * STRIDE_FACTOR;

  // Apply the offset
  match += offset;

  for (int stride = 1; stride <= blockDim.x; stride *= STRIDE_FACTOR)
  {
    if (threadIdx.x % stride == 0 && offset + i + stride < num)
      match[i] += match[i + stride];
    __syncthreads();
  }

  // Write result for this block to global memory
  int matches;
  if (threadIdx.x == 0 && offset < num && (matches = match[0]) > 0)
    atomicAdd(count, matches);
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

  // execute on GPU
  char *text_d, *str_d, *match_d;
  int *count1_d, *count2_d, count1_h, count2_h;

  int const matchDim = textDim - strDim + 1;

  CHECK(cudaMalloc(&text_d, sizeof(char) * (textDim)));
  CHECK(cudaMalloc(&str_d, sizeof(char) * (strDim)));
  CHECK(cudaMalloc(&match_d, sizeof(char) * (matchDim)));
  CHECK(cudaMalloc(&count1_d, sizeof(int)));
  CHECK(cudaMalloc(&count2_d, sizeof(int)));
  CHECK(cudaMemcpy(text_d, text, sizeof(char) * (textDim), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(str_d, str, sizeof(char) * (strDim), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(count1_d, 0, sizeof(int)));
  CHECK(cudaMemset(count2_d, 0, sizeof(int)));

  dim3 threadsPerBlock(BLOCKDIM);
  dim3 numOfBlocks((matchDim + BLOCKDIM - 1) / BLOCKDIM);
  dim3 numOfBlocksReduction((matchDim + BLOCKDIM * STRIDE_FACTOR - 1) / (BLOCKDIM * STRIDE_FACTOR));

  // Kernel 1
  findStringKernel<<<numOfBlocks, threadsPerBlock>>>(text_d, textDim, str_d, strDim, match_d);
  CHECK_KERNELCALL();
  countMatchesKernel<<<numOfBlocksReduction, threadsPerBlock>>>(match_d, matchDim, count1_d);
  CHECK_KERNELCALL();
  CHECK(cudaMemcpy(&count1_h, count1_d, sizeof(int), cudaMemcpyDeviceToHost));

  // Kernel 2
  size_t const sharedMemSize = (BLOCKDIM + strDim) * sizeof(char);
  findStringKernelSharedMem<<<numOfBlocks, threadsPerBlock, sharedMemSize>>>(text_d, textDim, str_d, strDim, match_d);
  CHECK_KERNELCALL();
  countMatchesKernel<<<numOfBlocksReduction, threadsPerBlock>>>(match_d, matchDim, count2_d);
  CHECK_KERNELCALL();
  CHECK(cudaMemcpy(&count2_h, count2_d, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(text_d));
  CHECK(cudaFree(str_d));
  CHECK(cudaFree(match_d));
  CHECK(cudaFree(count1_d));
  CHECK(cudaFree(count2_d));

  // print results
  printText(text, textDim);
  printText(str, strDim);
  printf("%d\n", count);
  printf("Kernel 1: %d, %s\n", count1_h, count == count1_h ? "OK" : "KO");
  printf("Kernel 2: %d, %s\n", count2_h, count == count2_h ? "OK" : "KO");

  free(text);
  free(str);
  free(match);

  return 0;
}
