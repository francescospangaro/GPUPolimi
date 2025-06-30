#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define CHECK(call)                                               \
    {                                                             \
        cudaError_t checkErr = (call);                            \
        if (checkErr != cudaSuccess)                              \
        {                                                         \
            printf("Failed: %s\n", cudaGetErrorString(checkErr)); \
            assert(checkErr == cudaSuccess);                      \
        }                                                         \
    }

#define CHECK_LAST() CHECK(cudaGetLastError())

__global__ void saxpy(int const n,
                      float const a,
                      int const *const __restrict__ x,
                      int *const __restrict__ y)
{
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        y[tid] = a * x[tid] + y[tid];
}

int main(int const argc, char **argv)
{
    int *x_h, *y_h;
    int *x_d, *y_d;

    constexpr int N = 1 << 20;
    const int blockSize = 256;
    const float multiplier = .14f;

    CHECK(cudaMalloc(&x_d, N * sizeof(*x_d)));
    CHECK(cudaMalloc(&y_d, N * sizeof(*y_d)));

    CHECK(cudaMallocHost(&x_h, N * sizeof(*x_h)));
    CHECK(cudaMallocHost(&y_h, N * sizeof(*y_h)));

    for (int i = 0; i < N; ++i)
    {
        x_h[i] = i;
        y_h[i] = N - i;
    }

    CHECK(cudaMemcpy(x_d, x_h, N * sizeof(*x_h), cudaMemcpyHostToDevice));

    saxpy<<<(N + blockSize - 1) / blockSize, blockSize>>>(N, multiplier, x_d, y_d);
    CHECK_LAST();

    CHECK(cudaMemcpy(y_h, y_d, N * sizeof(*y_d), cudaMemcpyDeviceToHost));

    cudaFreeHost(x_h);
    cudaFreeHost(y_h);
    cudaFree(x_d);
    cudaFree(y_d);
    return 0;
}