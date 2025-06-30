#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

// CPU version of the scan kernel
void scan_cpu(const float* input, float* output, const int length) {
  output[0] = input[0];
  for (int i = 1; i < length; ++i) { output[i] = output[i - 1] + input[i]; }
}

#define DIM 512

int main() {
  std::vector<float> input(DIM);
  std::vector<float> output(DIM);
  for (int i = 0; i < DIM; ++i)
    input[i] = static_cast<float>(rand()) / RAND_MAX; // Random value between 0 and 1 }

  // CPU version
  scan_cpu(input.data(), output.data(), input.size());

  return 0;
}
