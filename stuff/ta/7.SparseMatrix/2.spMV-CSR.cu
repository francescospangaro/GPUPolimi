#include <cstddef>
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
#define NUM_ROW      5
#define NUM_COLUMN   5
#define NUM_NON_ZERO 10

#define BLOCK_DIM 128

struct SparseMatrixCSR {
  int* row_indices;
  int* column_indices;
  double* values;
  int num_rows;
  int num_columns;
  int rows_indexes_length;
  int num_elements;

  SparseMatrixCSR(const int num_rows, const int num_columns, const int num_non_zeros) {
    this->num_rows    = num_rows;
    this->num_columns = num_columns;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> row_dist(0, num_rows - 1);
    std::uniform_int_distribution<> col_dist(0, num_columns - 1);
    std::uniform_real_distribution<> val_dist(MIN_VALUE, MAX_VALUE);

    int last_row = 0;
    std::vector<int> row_indices;
    std::vector<int> column_indices;
    std::vector<double> values;

    row_indices.push_back(0);
    for (int i = 0; i < num_non_zeros; ++i) {
      int row    = row_dist(gen);
      int col    = col_dist(gen);
      double val = val_dist(gen);

      column_indices.push_back(col);
      values.push_back(val);

      if (row != last_row) {
        for (int r = last_row + 1; r <= row; ++r) { row_indices.push_back(i); }
        last_row = row;
      }
    }
    while (last_row < num_rows) {
      row_indices.push_back(num_non_zeros);
      last_row++;
    }

    this->column_indices = new int[column_indices.size()];
    this->row_indices    = new int[row_indices.size()];
    this->values         = new double[values.size()];
    std::memcpy(this->column_indices, column_indices.data(), column_indices.size() * sizeof(int));
    std::memcpy(this->row_indices, row_indices.data(), row_indices.size() * sizeof(int));
    std::memcpy(this->values, values.data(), values.size() * sizeof(double));
    this->rows_indexes_length = row_indices.size();
    this->num_elements        = column_indices.size();
  }

  ~SparseMatrixCSR() {
    delete this->column_indices;
    delete this->row_indices;
    delete this->values;
  }
};

struct SparseMatrixCSR_gpu {
  int* row_indices;
  int* column_indices;
  double* values;
  int num_rows;
  int num_columns;

  SparseMatrixCSR_gpu(const SparseMatrixCSR& A) {
    this->num_rows    = A.num_rows;
    this->num_columns = A.num_columns;
    // Allocate memory on GPU
    CHECK(cudaMalloc(&this->row_indices, A.rows_indexes_length * sizeof(int)));
    CHECK(cudaMalloc(&this->column_indices, A.num_elements * sizeof(int)));
    CHECK(cudaMalloc(&this->values, A.num_elements * sizeof(double)));

    CHECK(cudaMemcpy(this->row_indices,
                     A.row_indices,
                     A.rows_indexes_length * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->column_indices,
                     A.column_indices,
                     A.num_elements * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(this->values, A.values, A.num_elements * sizeof(double), cudaMemcpyHostToDevice));
  };

  ~SparseMatrixCSR_gpu() {
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

__global__ void SpMV_CSR_gpu(const int* __restrict__ row_indices,
                             const int* __restrict__ column_indices,
                             const double* __restrict__ values,
                             const int num_rows,
                             const int num_columns,
                             const double* __restrict__ x,
                             double* __restrict__ y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO
  if(row < num_rows) {
    double dotProduct = 0;
    for (int k = row_indices[row], row_end = row_indices[row + 1]; k < row_end; ++k) {
      int j = column_indices[k];
      dotProduct += values[k] * x[j];
    }
    y[row] += dotProduct;
  }
}

void SpMV_CSR(const SparseMatrixCSR* A, const double* x, double* y) {
  for (int i = 0; i < A->num_rows; ++i) {
    for (int k = A->row_indices[i]; k < A->row_indices[i + 1]; ++k) {
      int j = A->column_indices[k];
      y[i] += A->values[k] * x[j];
    }
  }
}

int main() {
  // Example sparse matrix in CSR format
  SparseMatrixCSR A{NUM_ROW, NUM_COLUMN, NUM_NON_ZERO};

  // Example dense vector
  std::vector<double> x(A.num_columns);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(MIN_VALUE, MAX_VALUE);

  for (int i = 0; i < A.num_columns; ++i) { x[i] = dist(gen); }

  // Result vector
  std::vector<double> cpu_result(A.num_rows);
  for (int i = 0; i < cpu_result.size(); ++i) { cpu_result[i] = 0.0; }

  // Perform sparse matrix-vector multiplication
  SpMV_CSR(&A, x.data(), cpu_result.data());

  // Allocate memory on GPU
  SparseMatrixCSR_gpu A_gpu{A};
  double *vector_output_gpu, *vector_input_gpu;
  CHECK(cudaMalloc(&vector_output_gpu, A.num_rows * sizeof(double)));
  CHECK(cudaMalloc(&vector_input_gpu, A.num_columns * sizeof(double)));
  CHECK(cudaMemcpy(vector_input_gpu, x.data(), A.num_columns * sizeof(double), cudaMemcpyHostToDevice));

  // Configure GPU kernel launch
  dim3 threads_per_block(BLOCK_DIM);
  dim3 blocks_per_grid((x.size() + BLOCK_DIM - 1) / BLOCK_DIM);

  // Launch GPU kernel
  SpMV_CSR_gpu<<<blocks_per_grid, threads_per_block>>>(A_gpu.row_indices,
                                                       A_gpu.column_indices,
                                                       A_gpu.values,
                                                       A_gpu.num_rows,
                                                       A_gpu.num_columns,
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
      std::cout << "Sparse Matrix Vector Multiplication CSR CPU and GPU are NOT equivalent!" << std::endl;
      std::cout << "Index: " << i << std::endl;
      std::cout << "CPU: " << cpu_result[i] << std::endl;
      std::cout << "GPU: " << gpu_result[i] << std::endl;
      return EXIT_FAILURE;
    }

  std::cout << "Sparse Matrix Vector Multiplication CSR CPU and GPU are equivalent!" << std::endl;

  CHECK(cudaFree(vector_input_gpu));
  CHECK(cudaFree(vector_output_gpu));

  return EXIT_SUCCESS;
}