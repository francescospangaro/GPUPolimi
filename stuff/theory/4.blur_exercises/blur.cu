#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                            \
  {                                                            \
    const cudaError_t err = (call);                            \
    if (err != cudaSuccess)                                    \
    {                                                          \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), \
             __FILE__, __LINE__);                              \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

#define CHECK_KERNELCALL()                                     \
  {                                                            \
    const cudaError_t err = cudaGetLastError();                \
    if (err != cudaSuccess)                                    \
    {                                                          \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), \
             __FILE__, __LINE__);                              \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

#define CHANNELS 3
#define OUT_FN_CPU "output.pgm"
#define BLURDIM 10

int save_ppm_image(const char *filename, unsigned char *image, unsigned int width, unsigned int height);
int save_pgm_image(const char *filename, unsigned char *image, unsigned int width, unsigned int height);
int load_ppm_image(const char *filename, unsigned char **image, unsigned int *width, unsigned int *height);
int load_pgm_image(const char *filename, unsigned char **image, unsigned int *width, unsigned int *height);
__global__ void rgb2gray(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height);
__global__ void blur(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height);

__global__ void rgb2gray(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height)
{
  unsigned char redValue, greenValue, blueValue, grayValue;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < width && i < height)
  {
    redValue = input[(i * width + j) * 3];
    greenValue = input[(i * width + j) * 3 + 1];
    blueValue = input[(i * width + j) * 3 + 2];
    grayValue = (unsigned char)(0.299 * redValue + 0.587 * greenValue + 0.114 * blueValue);
    output[i * width + j] = grayValue;
  }
}

__global__ void blur(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < width && i < height)
  {
    int count = 0;
    int sum = 0;
    for (int h = -BLURDIM; h <= BLURDIM; h++)
    {
      for (int k = -BLURDIM; k <= BLURDIM; k++)
      {
        int inBounds = (i + h >= 0) & (i + h < height) & (j + k >= 0) & (j + k < width);
        count += inBounds;
        sum += inBounds * input[inBounds * ((i + h) * width + (j + k))];
      }
    }

    output[i * width + j] = (float)sum / count;
  }
}

int main(int argc, char *argv[])
{
  // read arguments
  if (argc != 4)
  {
    printf("Please specify ppm input file name\n");
    return 0;
  }

  int err;
  int ret = 1;

  char *inputfile = argv[1];
  int blockdim_x = atoi(argv[2]);
  int blockdim_y = atoi(argv[3]);

  // load input image
  unsigned int height, width;
  unsigned char *input;
  err = load_ppm_image(inputfile, &input, &width, &height);
  if (err)
    return 1;

  int nPixels = width * height;
  unsigned char *input_d;
  cudaMalloc(&input_d, sizeof(unsigned char) * nPixels * CHANNELS);
  if (!input_d)
  {
    printf("Error with malloc\n");
    goto free_input;
  }

  cudaMemcpy(input_d, input, sizeof(unsigned char) * nPixels * CHANNELS, cudaMemcpyHostToDevice);

  // allocate memory for gray image
  unsigned char *gray;
  cudaMalloc(&gray, sizeof(unsigned char) * nPixels);
  if (!gray)
  {
    printf("Error with malloc\n");
    goto free_input_d;
  }

  // allocate memory for output image
  unsigned char *output_d;
  cudaMalloc(&output_d, sizeof(unsigned char) * nPixels);
  if (!output_d)
  {
    printf("Error with malloc\n");
    goto free_gray;
  }

  unsigned char *output;
  output = (unsigned char *)malloc(sizeof(unsigned char) * nPixels);
  if (!output)
  {
    printf("Error with malloc\n");
    goto free_output_d;
  }

  float time;
  {
    dim3 blocksPerGrid(ceilf(width / blockdim_x), ceilf(height / blockdim_y), 1);
    dim3 threadsPerBlock(blockdim_x, blockdim_y, 1);

    int dev;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&deviceProp, dev);

    if (threadsPerBlock.x <= 0 || threadsPerBlock.x > deviceProp.maxThreadsDim[0] ||
        threadsPerBlock.y <= 0 || threadsPerBlock.y > deviceProp.maxThreadsDim[1])
    {
      printf("Violated maximum sizes of a dimension of a block (0;%d] - (0:%d] - Specified values: %d %d\n",
             deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
             threadsPerBlock.x, threadsPerBlock.y);
      goto free_output;
    }

    if (threadsPerBlock.x * threadsPerBlock.y > deviceProp.maxThreadsPerBlock)
    {
      printf("Violated maximum number of threads per block (%d) - Specified value: %d\n",
             deviceProp.maxThreadsPerBlock, threadsPerBlock.x * threadsPerBlock.y);
      goto free_output;
    }

    // process image
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    CHECK(cudaEventRecord(start));
    rgb2gray<<<blocksPerGrid, threadsPerBlock>>>(input_d, gray, width, height);
    CHECK_KERNELCALL();
    blur<<<blocksPerGrid, threadsPerBlock>>>(gray, output_d, width, height);
    CHECK_KERNELCALL();

    CHECK(cudaEventRecord(end));
    CHECK(cudaMemcpy(output, output_d, nPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CHECK(cudaEventElapsedTime(&time, start, end));

    cudaEventDestroy(start);
    cudaEventDestroy(end);
  }
  printf("Took %fms\n", time);

  // save output image
  err = save_pgm_image(OUT_FN_CPU, output, width, height);
  if (err)
  {
    printf("Failed to save image\n");
    goto free_output;
  }

  ret = 0;
// cleanup
free_output:
  free(output);
free_output_d:
  cudaFree(output_d);
free_gray:
  cudaFree(gray);
free_input_d:
  cudaFree(input_d);
free_input:
  free(input);

  return ret;
}

int save_ppm_image(const char *filename, unsigned char *image, unsigned int width, unsigned int height)
{
  FILE *f; // output file handle

  // open the output file and write header info for PPM filetype
  f = fopen(filename, "wb");
  if (f == NULL)
  {
    fprintf(stderr, "Error opening 'output.ppm' output file\n");
    return -1;
  }
  fprintf(f, "P6\n");
  fprintf(f, "%d %d\n%d\n", width, height, 255);
  fwrite(image, sizeof(unsigned char), height * width * CHANNELS, f);
  fclose(f);
  return 0;
}

int save_pgm_image(const char *filename, unsigned char *image, unsigned int width, unsigned int height)
{
  FILE *f; // output file handle

  // open the output file and write header info for PPM filetype
  f = fopen(filename, "wb");
  if (f == NULL)
  {
    fprintf(stderr, "Error opening 'output.ppm' output file\n");
    return -1;
  }
  fprintf(f, "P5\n");
  fprintf(f, "%d %d\n%d\n", width, height, 255);
  fwrite(image, sizeof(unsigned char), height * width, f);
  fclose(f);
  return 0;
}

int load_ppm_image(const char *filename, unsigned char **image, unsigned int *width, unsigned int *height)
{
  FILE *f; // input file handle
  char temp[256];
  unsigned int s;

  // open the input file and write header info for PPM filetype
  f = fopen(filename, "rb");
  if (f == NULL)
  {
    fprintf(stderr, "Error opening '%s' input file\n", filename);
    return -1;
  }
  fscanf(f, "%s\n", temp);
  fscanf(f, "%d %d\n", width, height);
  fscanf(f, "%d\n", &s);

  *image = (unsigned char *)malloc(sizeof(unsigned char) * (*width) * (*height) * CHANNELS);
  if (*image)
    fread(*image, sizeof(unsigned char), (*width) * (*height) * CHANNELS, f);
  else
  {
    printf("Error with malloc\n");
    return -1;
  }

  fclose(f);
  return 0;
}

int load_pgm_image(const char *filename, unsigned char **image, unsigned int *width, unsigned int *height)
{
  FILE *f; // input file handle
  char temp[256];
  unsigned int s;

  // open the input file and write header info for PPM filetype
  f = fopen(filename, "rb");
  if (f == NULL)
  {
    fprintf(stderr, "Error opening '%s' input file\n", filename);
    return -1;
  }
  fscanf(f, "%s\n", temp);
  fscanf(f, "%d %d\n", width, height);
  fscanf(f, "%d\n", &s);

  *image = (unsigned char *)malloc(sizeof(unsigned char) * (*width) * (*height));
  if (*image)
    fread(*image, sizeof(unsigned char), (*width) * (*height), f);
  else
  {
    printf("Error with malloc\n");
    return -1;
  }

  fclose(f);
  return 0;
}
