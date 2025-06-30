#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>

#define BLOCK_QUEUE_SIZE 128

#define BLOCK_SIZE 128

#define MAX_FRONTIER_SIZE BLOCK_SIZE

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

void read_matrix(std::vector<int> &row_ptr,
                 std::vector<int> &col_ind,
                 std::vector<float> &values,
                 const std::string &filename,
                 int &num_rows,
                 int &num_cols,
                 int &num_vals);

void insertIntoFrontier(int val, int *frontier, int *frontier_size)
{
  frontier[*frontier_size] = val;
  *frontier_size = *frontier_size + 1;
}

inline void swap(int **ptr1, int **ptr2)
{
  int *tmp = *ptr1;
  *ptr1 = *ptr2;
  *ptr2 = tmp;
}

void BFS_sequential(const int source, const int *rowPointers, const int *destinations, int *distances)
{
  int **frontiers_host = (int **)malloc(2 * sizeof(int *));
  for (int i = 0; i < 2; i++)
    frontiers_host[i] = (int *)calloc(MAX_FRONTIER_SIZE, sizeof(int));
  int *currentFrontier = frontiers_host[0];
  int currentFrontierSize = 0;
  int *previousFrontier = frontiers_host[1];
  int previousFrontierSize = 0;
  insertIntoFrontier(source, previousFrontier, &previousFrontierSize);
  distances[source] = 0;
  while (previousFrontierSize > 0)
  {
    // visit all vertices on the previous frontier
    for (int f = 0; f < previousFrontierSize; f++)
    {
      const int currentVertex = previousFrontier[f];
      // check all outgoing edges
      for (int i = rowPointers[currentVertex]; i < rowPointers[currentVertex + 1]; ++i)
      {
        if (distances[destinations[i]] == -1)
        {
          // this vertex has not been visited yet
          insertIntoFrontier(destinations[i], currentFrontier, &currentFrontierSize);
          distances[destinations[i]] = distances[currentVertex] + 1;
        }
      }
    }
    swap(&currentFrontier, &previousFrontier);
    previousFrontierSize = currentFrontierSize;
    currentFrontierSize = 0;
  }
}

__global__ void BFS_Bqueue_kernel(const int *previousFrontier,
                                  const int *previousFrontierSize,
                                  int *currentFrontier,
                                  int *currentFrontierSize,
                                  const cudaTextureObject_t rowPointersTexture,
                                  const int *destinations,
                                  int *distances,
                                  int *visited)
{
  const int t = threadIdx.x + blockDim.x * blockIdx.x;
  if (t < *previousFrontierSize)
  {
    const int currentVertex = previousFrontier[t];
    // check all outgoing edges
    for (int i = tex1D<int>(rowPointersTexture, currentVertex); i < tex1D<int>(rowPointersTexture, currentVertex + 1); ++i)
    {
      bool alreadyVisited = atomicExch(&visited[destinations[i]], 1);
      if (!alreadyVisited)
      {
        // this vertex has not been visited yet
        currentFrontier[atomicAdd(currentFrontierSize, 1)] = destinations[i];
        // No need to CAS this, we are the only ones setting it thanks to visited
        distances[destinations[i]] = distances[currentVertex] + 1;
      }
    }
  }
  __syncthreads();
}

void BFS_host(const int source,
              int *distances,
              const int *rowPointers,
              const int *destinations,
              const int num_rows,
              const int num_vals)
{
  // Allocate frontiers
  int *dCurrentFrontier;
  int *dPreviousFrontier;
  int *dRowPointers, *dDestinations, *dDistances, *dVisited, *dCurrentFrontierSize, *dPreviousFrontierSize;

  // allocate texture memory
  cudaTextureObject_t rowPointersTexture;
  cudaArray *texArray = 0;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
  CHECK(cudaMallocArray(&texArray, &channelDesc, num_vals));
  CHECK(cudaMemcpy2DToArray(texArray,
                            0,
                            0,
                            rowPointers,
                            sizeof(int) * num_vals,
                            sizeof(int) * num_vals,
                            1,
                            cudaMemcpyHostToDevice));
  // setup tex parameters
  cudaResourceDesc tex_res;
  memset(&tex_res, 0, sizeof(cudaResourceDesc));
  tex_res.res.array.array = texArray;
  tex_res.resType = cudaResourceTypeArray;
  cudaTextureDesc texture_desc;
  memset(&texture_desc, 0, sizeof(cudaTextureDesc));
  texture_desc.normalizedCoords = false;         // access with normalized texture coordinates
  texture_desc.filterMode = cudaFilterModePoint; // linear interpolation?
  texture_desc.addressMode[0] = cudaAddressModeClamp;
  texture_desc.addressMode[1] = cudaAddressModeClamp;
  texture_desc.addressMode[2] = cudaAddressModeClamp;
  texture_desc.readMode = cudaReadModeElementType;
  CHECK(cudaCreateTextureObject(&rowPointersTexture, &tex_res, &texture_desc, NULL));

  CHECK(cudaMalloc(&dRowPointers, sizeof(int) * (num_rows + 1)));
  CHECK(cudaMalloc(&dDestinations, sizeof(int) * num_vals));
  CHECK(cudaMalloc(&dDistances, sizeof(int) * num_vals));
  CHECK(cudaMalloc(&dCurrentFrontier, sizeof(int) * MAX_FRONTIER_SIZE));
  CHECK(cudaMalloc(&dPreviousFrontier, sizeof(int) * MAX_FRONTIER_SIZE));
  CHECK(cudaMalloc(&dVisited, sizeof(int) * num_vals));
  CHECK(cudaMalloc(&dCurrentFrontierSize, sizeof(int)));
  CHECK(cudaMalloc(&dPreviousFrontierSize, sizeof(int)));

  CHECK(cudaMemcpy(dRowPointers, rowPointers, sizeof(int) * (num_rows + 1), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dDestinations, destinations, sizeof(int) * num_vals, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dDistances, distances, sizeof(int) * num_vals, cudaMemcpyHostToDevice));

  CHECK(cudaMemset(dVisited, 0, sizeof(int) * num_vals));

  int hPreviousFrontierSize = 1;
  CHECK(cudaMemcpy(dPreviousFrontier, &source, sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dPreviousFrontierSize, &hPreviousFrontierSize, sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dDistances + source, 0, sizeof(int)));
  CHECK(cudaMemset(dVisited + source, 1, sizeof(int)));
  while (hPreviousFrontierSize > 0)
  {
    const int numBlocks = (hPreviousFrontierSize - 1) / BLOCK_SIZE + 1;
    BFS_Bqueue_kernel<<<numBlocks, BLOCK_SIZE>>>(dPreviousFrontier,
                                                 dPreviousFrontierSize,
                                                 dCurrentFrontier,
                                                 dCurrentFrontierSize,
                                                 rowPointersTexture,
                                                 dDestinations,
                                                 dDistances,
                                                 dVisited);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());
    // TODO Missing data movement to synchronize data
    swap(&dCurrentFrontier, &dPreviousFrontier);
    // TODO Missing data movement to prepare for the next iteration
    CHECK(cudaMemcpy(dPreviousFrontierSize, dCurrentFrontierSize, sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemset(dCurrentFrontierSize, 0, sizeof(int)));
    CHECK(cudaMemcpy(&hPreviousFrontierSize, dPreviousFrontierSize, sizeof(int), cudaMemcpyDeviceToHost));
  }
  CHECK(cudaMemcpy(distances, dDistances, num_vals * sizeof(int), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(dRowPointers));
  CHECK(cudaFree(dDestinations));
  CHECK(cudaFree(dDistances));
  CHECK(cudaFree(dCurrentFrontier));
  CHECK(cudaFree(dPreviousFrontier));
  CHECK(cudaFree(dVisited));
  CHECK(cudaFree(dCurrentFrontierSize));
  CHECK(cudaFree(dPreviousFrontierSize));
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("Usage: ./exec matrix_file source\n");
    return EXIT_FAILURE;
  }

  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<float> values;
  int num_rows, num_cols, num_vals;

  // int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
  // float *values;

  const std::string filename{argv[1]};
  // The node starts from 1 but array starts from 0
  const int source = atoi(argv[2]) - 1;

  read_matrix(row_ptr, col_ind, values, filename, num_rows, num_cols, num_vals);

  // Initialize dist to -1
  std::vector<int> dist(num_vals);
  for (int i = 0; i < num_vals; i++)
  {
    dist[i] = -1;
  }
  std::vector<int> dist_gpu(num_vals);
  for (int i = 0; i < num_vals; i++)
  {
    dist_gpu[i] = -1;
  }
  // Compute on the gpu
  BFS_host(source, dist_gpu.data(), row_ptr.data(), col_ind.data(), num_rows, num_vals);

  // Compute in sw
  BFS_sequential(source, row_ptr.data(), col_ind.data(), dist.data());

  for (size_t i = 0; i < num_vals; i++)
  {
    if (dist[i] != dist_gpu[i])
    {
      printf(" ERROR! INDEX  %zu CPU %d GPU %d\n", i, dist[i], dist_gpu[i]);
      return EXIT_FAILURE;
    }
  }

  printf("ALL RESULTS CORRECT\n");

  return EXIT_SUCCESS;
}

void read_matrix(std::vector<int> &row_ptr,
                 std::vector<int> &col_ind,
                 std::vector<float> &values,
                 const std::string &filename,
                 int &num_rows,
                 int &num_cols,
                 int &num_vals)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "File cannot be opened!\n";
    throw std::runtime_error("File cannot be opened");
  }

  // Get number of rows, columns, and non-zero values
  file >> num_rows >> num_cols >> num_vals;

  row_ptr.resize(num_rows + 1);
  col_ind.resize(num_vals);
  values.resize(num_vals);

  // Collect occurrences of each row for determining the indices of row_ptr
  std::vector<int> row_occurrences(num_rows, 0);

  int row, column;
  float value;
  while (file >> row >> column >> value)
  {
    // Subtract 1 from row and column indices to match C format
    row--;
    column--;

    row_occurrences[row]++;
  }

  // Set row_ptr
  int index = 0;
  for (int i = 0; i < num_rows; i++)
  {
    row_ptr[i] = index;
    index += row_occurrences[i];
  }
  row_ptr[num_rows] = num_vals;

  // Reset the file stream to read again from the beginning
  file.clear();
  file.seekg(0, std::ios::beg);

  // Read the first line again to skip it
  file >> num_rows >> num_cols >> num_vals;

  std::fill(col_ind.begin(), col_ind.end(), -1);

  int i = 0;
  while (file >> row >> column >> value)
  {
    row--;
    column--;

    // Find the correct index (i + row_ptr[row]) using both row information and an index i
    while (col_ind[i + row_ptr[row]] != -1)
    {
      i++;
    }
    col_ind[i + row_ptr[row]] = column;
    values[i + row_ptr[row]] = value;
    i = 0;
  }
}
