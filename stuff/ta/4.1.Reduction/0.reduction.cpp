#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

// CPU version of the reduction kernel
float reduce_cpu(const float* data, const int length) {
  float sum = 0;
  for (int i = 0; i < length; i++) { sum += data[i]; }
  return sum;
}

#define DIM 512

int main() {
  std::vector<float> data(DIM);
  for (int i = 0; i < DIM; ++i)
    data[i] = static_cast<float>(rand()) / RAND_MAX; // Random value between 0 and 1 }

  // CPU version
  float sum_cpu = reduce_cpu(data.data(), data.size());

  std::cout << "Reduction CPU is " << sum_cpu << std::endl;

  return 0;
}
