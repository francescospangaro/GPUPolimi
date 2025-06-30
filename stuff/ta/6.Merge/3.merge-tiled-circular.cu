#include <cstdlib>
#include <iostream>
#include <vector>

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

#define SIZE_A 20000
#define SIZE_B 12768
#define SIZE_C (SIZE_A + SIZE_B)

#define BLOCK_DIM 128
#define GRID_DIM 32
#define TILE_SIZE 1024

__device__ __forceinline__ int gpu_ceil(const int a, const int b) { return ceil((double)a / b); }
__device__ __forceinline__ int gpu_min(const int a, const int b) { return min(a, b); }
__device__ __forceinline__ int gpu_max(const int a, const int b) { return max(a, b); }

// Function for CPU and GPU merge sequentia computation
__device__ __host__ void
merge_sequential(const int *A, const int dim_A, const int *B, const int dim_B, int *C)
{
  int i = 0;
  int j = 0;
  int k = 0;

  while ((i < dim_A) && (j < dim_B))
  {
    if (A[i] <= B[j])
      C[k++] = A[i++];
    else
      C[k++] = B[j++];
  }
  if (i == dim_A)
  {
    while (j < dim_B)
    {
      C[k++] = B[j++];
    }
  }
  else
  {
    while (i < dim_A)
    {
      C[k++] = A[i++];
    }
  }
}

__device__ void merge_sequential_circular(const int *__restrict__ A,
                                          const int m,
                                          const int *__restrict__ B,
                                          const int n,
                                          int *__restrict__ C,
                                          const int A_S_start,
                                          const int B_S_start,
                                          const int tile_size)
{
  int i = 0; // Virtual index into A
  int j = 0; // Virtual index into B
  int k = 0; // virtual index into C

  while ((i < m) && (j < n))
  {
    int i_cir = (A_S_start + i) % tile_size;
    int j_cir = (B_S_start + j) % tile_size;

    // TODO
    if (A[i_cir] <= B[j_cir])
    {
      C[k++] = A[i_cir];
      i++;
    }
    else
    {
      C[k++] = B[j_cir];
      j++;
    }
  }

  if (i == m)
  { // Done with A[] handle remaining B[]
    // TODO
    for (; j < n; ++j)
      C[k++] = B[(B_S_start + j) % tile_size];
  }
  else
  { // Done with B[], handle remaining A[]
    // TODO
    for (; i < m; ++i)
      C[k++] = A[(A_S_start + i) % tile_size];
  }
}

// Function to compute the co-rank
__device__ int
co_rank(const int k, const int *__restrict__ A, const int m, const int *__restrict__ B, const int n)
{
  int i = gpu_min(k, m); // min
  int j = k - i;
  int i_low = gpu_max(0, (k - n)); // max
  int j_low = gpu_max(0, (k - m));
  int delta;
  bool active = true;
  while (active)
  {
    // TODO checks for binary search of the co-rank
    if (i > 0 && j < n && A[i - 1] > B[j])
    {
      delta = (i - i_low + 1) / 2;
      j_low = j;
      j = j + delta;
      i = i - delta;
      continue;
    }

    if (j > 0 && i < m && B[j - 1] >= A[i])
    {
      delta = (j - j_low + 1) / 2;
      i_low = i;
      i = i + delta;
      j = j - delta;
      continue;
    }

    active = false;
  }
  return i;
}

// Function to compute the co-rank circular version
__device__ int co_rank_circular(const int k,
                                const int *__restrict__ A,
                                const int m,
                                const int *__restrict__ B,
                                const int n,
                                const int A_S_start,
                                const int B_S_start,
                                const int tile_size)
{
  int i = gpu_min(k, m); // min
  int j = k - i;
  int i_low = gpu_max(0, (k - n)); // max
  int j_low = gpu_max(0, (k - m));
  int delta;
  bool active = true;
  while (active)
  {
    // TODO checks for binary search of the co-rank
    const int i_cir = (A_S_start + i) % tile_size;
    const int i_m_1_cir = (A_S_start + i - 1) % tile_size;
    const int j_cir = (B_S_start + j) % tile_size;
    const int j_m_1_cir = (B_S_start + j - 1) % tile_size;

    if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir])
    {
      delta = (i - i_low + 1) / 2;
      j_low = j;
      j = j + delta;
      i = i - delta;
      continue;
    }

    if (j > 0 && i < m && B[j_m_1_cir] >= A[i_cir])
    {
      delta = (j - j_low + 1) / 2;
      i_low = i;
      i = i + delta;
      j = j - delta;
      continue;
    }

    active = false;
  }
  return i;
}

// Function for GPU merge computation
__global__ void merge_tiled_gpu(const int *__restrict__ A,
                                const int dim_A,
                                const int *__restrict__ B,
                                const int dim_B,
                                int *__restrict__ C)
{
  __shared__ int A_s[TILE_SIZE];
  __shared__ int B_s[TILE_SIZE];

  const int dim_C = dim_A + dim_B;

  const int C_curr = blockIdx.x * gpu_ceil(dim_C, gridDim.x);
  const int C_next = gpu_min((blockIdx.x + 1) * gpu_ceil(dim_C, gridDim.x), dim_C);

  if (threadIdx.x == 0)
  {
    A_s[0] = co_rank(C_curr, A, dim_A, B, dim_B);
    A_s[1] = co_rank(C_next, A, dim_A, B, dim_B);
  }
  __syncthreads();

  const int A_curr = A_s[0];
  const int A_next = A_s[1];
  const int B_curr = C_curr - A_curr;
  const int B_next = C_next - A_next;
  __syncthreads();

  int counter = 0;
  const int C_length = C_next - C_curr;
  const int A_length = A_next - A_curr;
  const int B_length = B_next - B_curr;
  const int total_iteration = gpu_ceil(C_length, TILE_SIZE);
  int C_completed = 0;
  int A_consumed = 0;
  int B_consumed = 0;
  int A_read = 0;
  int B_read = 0;
  int A_S_start = 0;
  int B_S_start = 0;
  int A_S_consumed = TILE_SIZE; // In the first iteration, fill the TILE_SIZE
  int B_S_consumed = TILE_SIZE; // In the first iteration, fill the tile_size

  while (counter < total_iteration)
  {
    // Portion of the computation remaining to be done
    const int C_remaining = C_length - C_completed;
    const int C_remaining_tile = gpu_min(TILE_SIZE, C_remaining);
    const int A_remaining = A_length - A_consumed;
    const int A_remaining_tile = gpu_min(TILE_SIZE, A_remaining);
    const int B_remaining = B_length - B_consumed;
    const int B_remaining_tile = gpu_min(TILE_SIZE, B_remaining);

    // Loading (refilling) A_S_consumed elements into A_S
    // TODO
    for (int i = threadIdx.x; i < A_S_consumed; i += blockDim.x)
    {
      if (i < A_remaining)
        A_s[(A_S_start + (TILE_SIZE - A_S_consumed) + i) % TILE_SIZE] = A[A_curr + i + A_read];
    }
    A_read += gpu_min(A_remaining, A_S_consumed);

    // Loading B_S_consumed elements into B_S
    // TODO
    for (int i = threadIdx.x; i < B_S_consumed; i += blockDim.x)
    {
      if (i < B_remaining)
        B_s[(B_S_start + (TILE_SIZE - B_S_consumed) + i) % TILE_SIZE] = B[B_curr + i + B_read];
    }
    B_read += gpu_min(B_remaining, B_S_consumed);
    __syncthreads();

    // Get sections of A, B, and C
    // TODO
    const size_t k_curr = gpu_min(threadIdx.x * (TILE_SIZE / blockDim.x), C_length - C_completed);
    const size_t k_next = gpu_min((threadIdx.x + 1) * (TILE_SIZE / blockDim.x), C_length - C_completed);

    const size_t i_curr = co_rank_circular(k_curr, A_s, A_remaining_tile, B_s, B_remaining_tile, A_S_start, B_S_start, TILE_SIZE);
    const size_t i_next = co_rank_circular(k_next, A_s, A_remaining_tile, B_s, B_remaining_tile, A_S_start, B_S_start, TILE_SIZE);

    const size_t j_curr = k_curr - i_curr;
    const size_t j_next = k_next - i_next;

    // Do merge in parallel
    // TODO
    merge_sequential_circular(A_s, i_next - i_curr,
                              B_s, j_next - j_curr,
                              &C[C_curr + C_completed + k_curr],
                              A_S_start + i_curr,
                              B_S_start + j_curr,
                              TILE_SIZE);

    // Update counter and number of computed elements
    // TODO
    counter++;
    A_S_consumed = co_rank_circular(C_remaining_tile, A_s, A_remaining_tile, B_s, B_remaining_tile, A_S_start, B_S_start, TILE_SIZE);
    B_S_consumed = C_remaining_tile - A_S_consumed;
    A_consumed += A_S_consumed;
    C_completed += C_remaining_tile;
    B_consumed = C_completed - A_consumed;

    A_S_start = A_S_start + A_S_consumed;
    if (A_S_start >= TILE_SIZE)
      A_S_start = A_S_start - TILE_SIZE;

    B_S_start = B_S_start + B_S_consumed;
    if (B_S_start >= TILE_SIZE)
      B_S_start = B_S_start - TILE_SIZE;

    __syncthreads();
  }
}

// simple sort, just to sort the randomly generated arrays
void sort_array(int *arr, const int dim)
{
  int i, j;

  for (i = 0; i < dim; i++)
  {
    for (j = i + 1; j < dim; j++)
    {
      if (arr[j] < arr[i])
      {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
      }
    }
  }
}

int main()
{
  static_assert((SIZE_C / GRID_DIM) % TILE_SIZE == 0, "Total size should be a multiple of TILE_SIZE");
  // Generate random input data
  std::vector<int> A_host(SIZE_A);
  std::vector<int> B_host(SIZE_B);
  for (int i = 0; i < SIZE_A; ++i)
  {
    A_host[i] = rand() % 1000;
  }
  for (int i = 0; i < SIZE_B; ++i)
  {
    B_host[i] = rand() % 1000;
  }
  sort_array(A_host.data(), SIZE_A);
  sort_array(B_host.data(), SIZE_B);

  // Compute merge on CPU
  std::vector<int> cpu_result(SIZE_C);
  merge_sequential(A_host.data(), SIZE_A, B_host.data(), SIZE_B, cpu_result.data());

  // Allocate memory on GPU
  int *A_device, *B_device, *C_device;
  CHECK(cudaMalloc(&A_device, SIZE_A * sizeof(int)));
  CHECK(cudaMalloc(&B_device, SIZE_B * sizeof(int)));
  CHECK(cudaMalloc(&C_device, SIZE_C * sizeof(int)));
  CHECK(cudaMemcpy(A_device, A_host.data(), SIZE_A * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B_device, B_host.data(), SIZE_B * sizeof(int), cudaMemcpyHostToDevice));

  // Configure GPU kernel launch
  dim3 threads_per_block(BLOCK_DIM);
  dim3 blocks_per_grid(GRID_DIM);
  // Launch GPU kernel
  merge_tiled_gpu<<<blocks_per_grid, threads_per_block>>>(A_device, SIZE_A, B_device, SIZE_B, C_device);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  std::vector<int> gpu_result(SIZE_C);
  CHECK(cudaMemcpy(gpu_result.data(), C_device, SIZE_C * sizeof(int), cudaMemcpyDeviceToHost));

  // Compare CPU and GPU results
  for (int i = 0; i < SIZE_C; ++i)
    if (cpu_result[i] != gpu_result[i])
    {
      std::cout << "Merge CPU and GPU are NOT equivalent!" << std::endl;
      std::cout << "Index: " << i << std::endl;
      std::cout << "CPU: " << cpu_result[i] << std::endl;
      std::cout << "GPU: " << gpu_result[i] << std::endl;
      return EXIT_FAILURE;
    }

  std::cout << "Merge CPU and GPU are equivalent!" << std::endl;

  // Free memory
  CHECK(cudaFree(A_device));
  CHECK(cudaFree(B_device));
  CHECK(cudaFree(C_device));

  return EXIT_SUCCESS;
}
