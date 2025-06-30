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

// #define linearize(x, y, DIM_X, DIM_Y) (x * DIM_Y + y)
#define linearize(x, y, DIM_X, DIM_Y) (x + y * DIM_X)

#define MIN_VALUE        1.0
#define MAX_VALUE        10.0
#define NUM_ROW          5
#define NUM_COLUMN       5
#define MAX_NON_ZERO_ROW 15

#define BLOCK_DIM 128

struct SparseMatrixELL {
  int* column_indices;
  double* values;
  int num_rows;
  int max_non_zero_row;

  SparseMatrixELL(const int num_rows, const int num_columns, const int max_non_zero_row) {
    this->num_rows         = num_rows;
    this->max_non_zero_row = max_non_zero_row;
    this->column_indices   = new int[max_non_zero_row * num_rows];
    this->values           = new double[max_non_zero_row * num_rows];
    std::memset(this->column_indices, 0, max_non_zero_row * num_rows * sizeof(int));
    std::memset(this->values, 0, max_non_zero_row * num_rows * sizeof(double));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> non_zero_per_row(0, max_non_zero_row);
    std::uniform_int_distribution<> col_dist(0, num_columns - 1);
    std::uniform_real_distribution<> val_dist(MIN_VALUE, MAX_VALUE);

    for (int row = 0; row < num_rows; ++row) {
      const int non_zero = non_zero_per_row(gen);
      for (int i = 0; i < non_zero; ++i) {
        const int col                                                 = col_dist(gen);
        const double val                                              = val_dist(gen);
        column_indices[linearize(row, i, num_rows, max_non_zero_row)] = col;
        values[linearize(row, i, num_rows, max_non_zero_row)]         = val;
      }
    }
  }

  ~SparseMatrixELL() {
    delete this->column_indices;
    delete this->values;
  }
};

struct SparseMatrixELL_gpu {
  int* column_indices;
  double* values;
  int num_rows;
  int max_non_zero_row;

  SparseMatrixELL_gpu(const SparseMatrixELL& A) {
    this->num_rows         = A.num_rows;
    this->max_non_zero_row = A.max_non_zero_row;

    // Allocate memory on GPU
    CHECK(cudaMalloc(&this->column_indices, A.max_non_zero_row * A.num_rows * sizeof(int)));
    CHECK(cudaMalloc(&this->values, A.max_non_zero_row * A.num_rows * sizeof(double)));

    CHECK(cudaMemcpy(this->column_indices,
                     A.column_indices,
                     A.max_non_zero_row * A.num_rows * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->values,
                     A.values,
                     A.max_non_zero_row * A.num_rows * sizeof(double),
                     cudaMemcpyHostToDevice));
  };

  ~SparseMatrixELL_gpu() {
    if (this->column_indices != nullptr) {
      CHECK(cudaFree(this->column_indices));
      this->column_indices = nullptr;
    }
    if (this->values != nullptr) {
      CHECK(cudaFree(this->values));
      this->values = nullptr;
    }
  }
};

void SpMV_ELL(const SparseMatrixELL* A, const double* x, double* y) {
  for (int i = 0; i < A->num_rows; ++i) {
    for (int k = 0; k < A->max_non_zero_row; ++k) {
      const int elementIndex = linearize(i, k, A->num_rows, A->max_non_zero_row);
      int j                  = A->column_indices[elementIndex];
      y[i] += A->values[elementIndex] * x[j];
    }
  }
}

__global__ void SpMV_ELL_gpu(const int* __restrict__ column_indices,
                             const double* __restrict__ values,
                             const int num_rows,
                             const int max_non_zero_row,
                             const double* __restrict__ x,
                             double* __restrict__ y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO
  if(row < num_rows) {
    double dotProduct = 0;
    for (int k = 0; k < max_non_zero_row; ++k) {
      const int elementIndex = linearize(row, k, num_rows, max_non_zero_row);
      dotProduct += values[elementIndex] * x[column_indices[elementIndex]];
    }
    y[row] += dotProduct;
  }
}

int main() {
  // Example sparse matrix in ELL format
  SparseMatrixELL A{NUM_ROW, NUM_COLUMN, MAX_NON_ZERO_ROW};

  // Example dense vector
  std::vector<double> x(NUM_COLUMN);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(MIN_VALUE, MAX_VALUE);

  for (int i = 0; i < NUM_COLUMN; ++i) { x[i] = dist(gen); }

  // Result vector
  std::vector<double> cpu_result(A.num_rows);
  for (int i = 0; i < cpu_result.size(); ++i) { cpu_result[i] = 0.0; }

  // Perform sparse matrix-vector multiplication
  SpMV_ELL(&A, x.data(), cpu_result.data());

  // Allocate memory on GPU
  SparseMatrixELL_gpu A_gpu{A};
  double *vector_output_gpu, *vector_input_gpu;
  CHECK(cudaMalloc(&vector_output_gpu, A.num_rows * sizeof(double)));
  CHECK(cudaMalloc(&vector_input_gpu, NUM_COLUMN * sizeof(double)));
  CHECK(cudaMemcpy(vector_input_gpu, x.data(), NUM_COLUMN * sizeof(double), cudaMemcpyHostToDevice));

  // Configure GPU kernel launch
  dim3 threads_per_block(BLOCK_DIM);
  dim3 blocks_per_grid((x.size() + BLOCK_DIM - 1) / BLOCK_DIM);

  // Launch GPU kernel
  SpMV_ELL_gpu<<<blocks_per_grid, threads_per_block>>>(A_gpu.column_indices,
                                                       A_gpu.values,
                                                       A_gpu.num_rows,
                                                       A_gpu.max_non_zero_row,
                                                       vector_input_gpu,
                                                       vector_output_gpu);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  std::vector<double> gpu_result(A.num_rows);
  CHECK(
      cudaMemcpy(gpu_result.data(), vector_output_gpu, A.num_rows * sizeof(double), cudaMemcpyDeviceToHost));

  // Compare CPU and GPU results
  for (int i = 0; i < A.num_rows; ++i)
    if (std::abs(cpu_result[i] - gpu_result[i]) > 1e-3) {
      std::cout << "Sparse Matrix Vector Multiplication ELL CPU and GPU are NOT equivalent!" << std::endl;
      std::cout << "Index: " << i << std::endl;
      std::cout << "CPU: " << cpu_result[i] << std::endl;
      std::cout << "GPU: " << gpu_result[i] << std::endl;
      return EXIT_FAILURE;
    }

  std::cout << "Sparse Matrix Vector Multiplication ELL CPU and GPU are equivalent!" << std::endl;

  CHECK(cudaFree(vector_input_gpu));
  CHECK(cudaFree(vector_output_gpu));

  return EXIT_SUCCESS;
}