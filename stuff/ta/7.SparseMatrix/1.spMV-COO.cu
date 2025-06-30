#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <random>
#include <vector>

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define MIN_VALUE    1.0
#define MAX_VALUE    10.0
#define NUM_ROW      25
#define NUM_COLUMN   25
#define NUM_ELEMENTS 15

#define BLOCK_DIM 128

struct SparseMatrixCOO {
  int* row_indices;
  int* column_indices;
  float* values;
  int num_elements;

  SparseMatrixCOO(const int num_rows, const int num_columns, const int num_elements) {
    this->num_elements = num_elements;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> row_dist(0, num_rows - 1);
    std::uniform_int_distribution<> col_dist(0, num_columns - 1);
    std::uniform_real_distribution<> val_dist(MIN_VALUE, MAX_VALUE);

    std::vector<int> row_indices;
    std::vector<int> column_indices;
    std::vector<float> values;

    row_indices.push_back(0);
    for (int i = 0; i < num_elements; ++i) {
      int row    = row_dist(gen);
      int col    = col_dist(gen);
      float val = val_dist(gen);

      column_indices.push_back(col);
      row_indices.push_back(row);
      values.push_back(val);
    }

    this->column_indices = new int[column_indices.size()];
    this->row_indices    = new int[row_indices.size()];
    this->values         = new float[values.size()];
    std::memcpy(this->column_indices, column_indices.data(), column_indices.size() * sizeof(int));
    std::memcpy(this->row_indices, row_indices.data(), row_indices.size() * sizeof(int));
    std::memcpy(this->values, values.data(), values.size() * sizeof(float));
  }

  ~SparseMatrixCOO() {
    delete this->column_indices;
    delete this->row_indices;
    delete this->values;
  }
};

struct SparseMatrixCOO_gpu {
  int* row_indices;
  int* column_indices;
  float* values;
  int num_elements;

  SparseMatrixCOO_gpu(const SparseMatrixCOO& A) {
    this->num_elements = A.num_elements;
    // Allocate memory on GPU
    CHECK(cudaMalloc(&this->row_indices, A.num_elements * sizeof(int)));
    CHECK(cudaMalloc(&this->column_indices, A.num_elements * sizeof(int)));
    CHECK(cudaMalloc(&this->values, A.num_elements * sizeof(float)));

    CHECK(cudaMemcpy(this->row_indices, A.row_indices, A.num_elements * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->column_indices,
                     A.column_indices,
                     A.num_elements * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->values, A.values, A.num_elements * sizeof(float), cudaMemcpyHostToDevice));
  };

  ~SparseMatrixCOO_gpu() {
    if (this->column_indices != nullptr) {
      CHECK(cudaFree(this->column_indices));
      this->column_indices = nullptr;
    }
    if (this->row_indices != nullptr) {
      CHECK(cudaFree(this->row_indices));
      this->row_indices = nullptr;
    }
    if (this->values != nullptr) {
      CHECK(cudaFree(this->values));
      this->values = nullptr;
    }
  }
};

void SpMV_COO(const SparseMatrixCOO* A, const float* x, float* y) {
  for (int element = 0; element < A->num_elements; ++element) {
    const int column = A->column_indices[element];
    const int row    = A->row_indices[element];
    y[row] += A->values[element] * x[column];
  }
}

__global__ void SpMV_COO_gpu(const int* __restrict__ row_indices,
                             const int* __restrict__ column_indices,
                             const float* __restrict__ values,
                             const int num_elements,
                             const float* x,
                             float* y) {
  // TODO
  int element = blockIdx.x * blockDim.x + threadIdx.x;
  for(; element < num_elements; element += blockDim.x * gridDim.x) {
      const int column = column_indices[element];
      const int row = row_indices[element];
      atomicAdd(&y[row], values[element] * x[column]);
  }
}

int main() {
  // Example sparse matrix in COO format
  SparseMatrixCOO A{NUM_ROW, NUM_COLUMN, NUM_ELEMENTS};

  // Example dense vector
  std::vector<float> x(NUM_COLUMN);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(MIN_VALUE, MAX_VALUE);

  for (int i = 0; i < NUM_COLUMN; ++i) { x[i] = dist(gen); }

  // Result vector
  std::vector<float> cpu_result(NUM_ROW);
  for (int i = 0; i < cpu_result.size(); ++i) { cpu_result[i] = 0.0; }

  // Perform sparse matrix-vector multiplication
  SpMV_COO(&A, x.data(), cpu_result.data());

  // Allocate memory on GPU
  SparseMatrixCOO_gpu A_gpu{A};
  float *vector_output_gpu, *vector_input_gpu;
  CHECK(cudaMalloc(&vector_output_gpu, NUM_ROW * sizeof(float)));
  CHECK(cudaMalloc(&vector_input_gpu, NUM_COLUMN * sizeof(float)));
  CHECK(cudaMemcpy(vector_input_gpu, x.data(), NUM_COLUMN * sizeof(float), cudaMemcpyHostToDevice));

  // Configure GPU kernel launch
  dim3 threads_per_block(BLOCK_DIM);
  dim3 blocks_per_grid((x.size() + BLOCK_DIM - 1) / BLOCK_DIM);

  // Launch GPU kernel
  SpMV_COO_gpu<<<blocks_per_grid, threads_per_block>>>(A_gpu.row_indices,
                                                       A_gpu.column_indices,
                                                       A_gpu.values,
                                                       A.num_elements,
                                                       vector_input_gpu,
                                                       vector_output_gpu);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  std::vector<float> gpu_result(NUM_ROW);
  CHECK(cudaMemcpy(gpu_result.data(), vector_output_gpu, NUM_ROW * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare CPU and GPU results
  for (int i = 0; i < NUM_ROW; ++i)
    if (std::abs(cpu_result[i] - gpu_result[i]) > 1e-3) {
      std::cout << "Sparse Matrix Vector Multiplication COO CPU and GPU are NOT equivalent!" << std::endl;
      std::cout << "Index: " << i << std::endl;
      std::cout << "CPU: " << cpu_result[i] << std::endl;
      std::cout << "GPU: " << gpu_result[i] << std::endl;
      return EXIT_FAILURE;
    }

  std::cout << "Sparse Matrix Vector Multiplication COO CPU and GPU are equivalent!" << std::endl;

  CHECK(cudaFree(vector_input_gpu));
  CHECK(cudaFree(vector_output_gpu));

  return EXIT_SUCCESS;
}