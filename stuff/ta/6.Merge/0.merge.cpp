#include <cstdlib>
#include <iostream>
#include <vector>

#define SIZE_A 20000
#define SIZE_B 12768

// Function for CPU merge computation
void merge_cpu(const int *A, const int dim_A, const int *B, const int dim_B, int *C) {
  int i = 0;
  int j = 0;
  int k = 0;

  while ((i < dim_A) && (j < dim_B)) {
    if (A[i] <= B[j])
      C[k++] = A[i++];
    else
      C[k++] = B[j++];
  }
  if (i == dim_A) {
    while (j < dim_B) { C[k++] = B[j++]; }
  } else {
    while (i < dim_A) { C[k++] = A[i++]; }
  }
}

// simple sort, just to sort the randomly generated arrays
void sort_array(int *arr, const int dim) {
  int i, j;

  for (i = 0; i < dim; i++) {
    for (j = i + 1; j < dim; j++) {
      if (arr[j] < arr[i]) {
        int tmp = arr[i];
        arr[i]  = arr[j];
        arr[j]  = tmp;
      }
    }
  }
}

int main() {
  // Generate random input data
  std::vector<int> A_host(SIZE_A);
  std::vector<int> B_host(SIZE_B);
  for (int i = 0; i < SIZE_A; ++i) { A_host[i] = rand() % 1000; }
  for (int i = 0; i < SIZE_B; ++i) { B_host[i] = rand() % 1000; }
  sort_array(A_host.data(), SIZE_A);
  sort_array(B_host.data(), SIZE_B);

  // Compute merge on CPU
  std::vector<int> C_host(SIZE_A + SIZE_B);
  merge_cpu(A_host.data(), SIZE_A, B_host.data(), SIZE_B, C_host.data());

  return EXIT_SUCCESS;
}
