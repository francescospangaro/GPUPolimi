#include <stdio.h>
#include <assert.h>

#define TILE_DIM 32
#define BLOCK_ROWS 32
#define NUM_REPS 10

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

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms)
{
  bool passed = true;
  for (int i = 0; i < n; i++)
    if (res[i] != ref[i])
    {
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  if (passed)
    printf("%20.2f\n", 2 * n * sizeof(float) * 1e-6 * NUM_REPS / ms);
}

// simple copy kernel
// Used as reference case representing best effective bandwidth.
__global__ void copy(float *__restrict__ odata, const float *__restrict__ idata)
{
  const int x = blockIdx.x * TILE_DIM + threadIdx.x;
  const int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = idata[(y + j) * width + x];
}

// copy kernel using shared memory
// Also used as reference case, demonstrating effect of using shared memory.
__global__ void copySharedMem(float *__restrict__ odata, const float *__restrict__ idata)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];

  const int x = blockIdx.x * TILE_DIM + threadIdx.x;
  const int y = blockIdx.y * TILE_DIM + threadIdx.y;
  const int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] = idata[(y + j) * width + x];

  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
}

// naive transpose
// Simplest transpose; doesn't use shared memory.
// Global memory reads are coalesced but writes are not.
__global__ void transposeNaive(float *__restrict__ odata, const float *__restrict__ idata)
{
  const int x = blockIdx.x * TILE_DIM + threadIdx.x;
  const int y = blockIdx.y * TILE_DIM + threadIdx.y;
  const int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[x * width + (y + j)] = idata[(y + j) * width + x];
}

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts(read).
__global__ void transposeCoalesced(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded
// to avoid shared memory bank conflicts.
__global__ void transposeNoBankConflicts(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main(int argc, char **argv)
{
  const int nx = 4096;
  const int ny = 4096;
  const int mem_size = nx * ny * sizeof(float);

  dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  int devId = 0;
  if (argc > 1)
    devId = atoi(argv[1]);

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, devId));
  printf("\nDevice : %s\n", prop.name);
  printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
         nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

  CHECK(cudaSetDevice(devId));

  float *h_idata = (float *)malloc(mem_size);
  float *h_cdata = (float *)malloc(mem_size);
  float *h_tdata = (float *)malloc(mem_size);
  float *gold = (float *)malloc(mem_size);

  float *d_idata, *d_cdata, *d_tdata;
  CHECK(cudaMalloc(&d_idata, mem_size));
  CHECK(cudaMalloc(&d_cdata, mem_size));
  CHECK(cudaMalloc(&d_tdata, mem_size));

  // check parameters and calculate execution configuration
  if (nx % TILE_DIM || ny % TILE_DIM)
  {
    printf("nx and ny must be a multiple of TILE_DIM\n");
    return EXIT_FAILURE;
  }

  if (TILE_DIM % BLOCK_ROWS)
  {
    printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
    return EXIT_FAILURE;
  }

  // host
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      h_idata[j * nx + i] = j * nx + i;

  // correct result for error checking
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      gold[j * nx + i] = h_idata[i * nx + j];

  // device
  CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

  // events for timing
  cudaEvent_t startEvent, stopEvent;
  CHECK(cudaEventCreate(&startEvent));
  CHECK(cudaEventCreate(&stopEvent));
  float ms;

  // ------------
  // time kernels
  // ------------
  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

  // ----
  // copy
  // ----
  printf("%25s", "copy");
  CHECK(cudaMemset(d_cdata, 0, mem_size));
  // warm up
  copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  CHECK(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; i++)
    copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  CHECK(cudaEventRecord(stopEvent, 0));
  CHECK(cudaEventSynchronize(stopEvent));
  CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  CHECK(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
  postprocess(h_idata, h_cdata, nx * ny, ms);

  // -------------
  // copySharedMem
  // -------------
  printf("%25s", "shared memory copy");
  CHECK(cudaMemset(d_cdata, 0, mem_size));
  // warm up
  copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  CHECK(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; i++)
    copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
  CHECK(cudaEventRecord(stopEvent, 0));
  CHECK(cudaEventSynchronize(stopEvent));
  CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  CHECK(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
  postprocess(h_idata, h_cdata, nx * ny, ms);

  // --------------
  // transposeNaive
  // --------------
  printf("%25s", "naive transpose");
  CHECK(cudaMemset(d_tdata, 0, mem_size));
  // warmup
  transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  CHECK(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; i++)
    transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  CHECK(cudaEventRecord(stopEvent, 0));
  CHECK(cudaEventSynchronize(stopEvent));
  CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  CHECK(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  postprocess(gold, h_tdata, nx * ny, ms);

  // ------------------
  // transposeCoalesced
  // ------------------
  printf("%25s", "coalesced transpose");
  CHECK(cudaMemset(d_tdata, 0, mem_size));
  // warmup
  transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  CHECK(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; i++)
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  CHECK(cudaEventRecord(stopEvent, 0));
  CHECK(cudaEventSynchronize(stopEvent));
  CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  CHECK(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  postprocess(gold, h_tdata, nx * ny, ms);

  // ------------------------
  // transposeNoBankConflicts
  // ------------------------
  printf("%25s", "conflict-free transpose");
  CHECK(cudaMemset(d_tdata, 0, mem_size));
  // warmup
  transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  CHECK(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < NUM_REPS; i++)
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
  CHECK(cudaEventRecord(stopEvent, 0));
  CHECK(cudaEventSynchronize(stopEvent));
  CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  CHECK(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
  postprocess(gold, h_tdata, nx * ny, ms);

  // cleanup
  CHECK(cudaEventDestroy(startEvent));
  CHECK(cudaEventDestroy(stopEvent));
  CHECK(cudaFree(d_tdata));
  CHECK(cudaFree(d_cdata));
  CHECK(cudaFree(d_idata));
  free(h_idata);
  free(h_tdata);
  free(h_cdata);
  free(gold);
}
