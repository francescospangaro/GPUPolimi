/*
 * The kernel function 1 (mult) performs the multiplication of a vector by a scalar value.
 * The kernel function 2 (compare) receives two vectors of integers, called A and B,
 * together with the sizes sa and sb, and a third empty vector of integers, C, which
 * size is sa*sb.
 * For each pair A[i] and B[j], the function saves in C[i][j] value 1 if A[i] > B[j],
 * 0 otherwise (do consider that the function manages C as a linearized array).
 * The main function is a dummy program receiving in input sa and sb, populating randomly A
 * and B, invoking the above two functions and showing results.
 */

#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 32
#define MAXVAL 100
#define VALUE 10

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }


void printM(int *M, int numMRows, int numMColumns);
void compare(int *M, int *N, int dm, int dn, int *P);
void mult(int *V, int dim, int fatt, int *P);
__global__ void compareKernel(int *M, int *N, int dm, int dn, int *P);
__global__ void multKernel(int *V, int dim, int fatt, int *P);
__global__ void compareKernelShared(int *M, int *N, int dm, int dn, int *P);

// display a matrix on the screen
void printM(int *M, int numMRows, int numMColumns)
{
    int i, j;
    for (i = 0; i < numMRows; i++)
    {
        for (j = 0; j < numMColumns; j++)
            printf("%3d ", M[i * numMColumns + j]);
        printf("\n");
    }
    printf("\n");
}

// kernel function 1: vector per scalar multiplication
void mult(int *V, int dim, int fatt, int *P)
{
    int i;
    for (i = 0; i < dim; i++)
        P[i] = V[i] * fatt;
}

__global__ void multKernel(int *V, int dim, int fatt, int *P) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < dim){
        P[i] = V[i] * fatt;
    }
}

// kernel function 2: compare each element of M against any element of N
void compare(int *M, int *N, int dm, int dn, int *P)
{
    int i, j;
    for (i = 0; i < dm; i++)
        for (j = 0; j < dn; j++)
            P[i * dn + j] = (M[i] > N[j]);
}

__global__ void compareKernel(int *M, int *N, int dm, int dn, int *P) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < dm && j < dn){
        P[i * dn + j] = (M[i] > N[j]);
    }
}

__global__ void compareKernelShared(int *M, int *N, int dm, int dn, int *P) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ int tileM[BLOCKSIZE];
    __shared__ int tileN[BLOCKSIZE];

    if (j < dm && threadIdx.x == 0)
        tileM[threadIdx.y] = M[j];
    if (i < dn && threadIdx.y == 0)
        tileN[threadIdx.x] = N[i];
    __syncthreads();
    if(i < dm && j < dn){
        P[i * dn + j] = (tileM[threadIdx.y] > tileN[threadIdx.x]);
    }
}

int main(int argc, char **argv)
{
    int *A, *B, *A1, *B1, *C;
    int sa, sb;
    int i, j;

    // read arguments
    if (argc != 3)
    {
        printf("Please specify sizes of vectors A and B\n");
        return 0;
    }
    sa = atoi(argv[1]);
    sb = atoi(argv[2]);

    // allocate memory for the three vectors
    A = (int *)malloc(sizeof(int) * sa);
    if (!A)
    {
        printf("Error: malloc failed\n");
        return 1;
    }
    A1 = (int *)malloc(sizeof(int) * sa);
    if (!A1)
    {
        free(A);
        printf("Error: malloc failed\n");
        return 1;
    }
    B = (int *)malloc(sizeof(int) * sb);
    if (!B)
    {
        printf("Error: malloc failed\n");
        free(A);
        free(A1);
        return 1;
    }
    B1 = (int *)malloc(sizeof(int) * sb);
    if (!B1)
    {
        printf("Error: malloc failed\n");
        free(A);
        free(A1);
        free(B);
        return 1;
    }
    C = (int *)malloc(sizeof(int) * sa * sb);
    if (!C)
    {
        printf("Error: malloc failed\n");
        free(A);
        free(A1);
        free(B);
        free(B1);
        return 1;
    }
    // initialize input vectors A and B
    srand(0);
    for (i = 0; i < sa; i++)
        A[i] = rand() % MAXVAL;
    for (i = 0; i < sb; i++)
        B[i] = rand() % MAXVAL;

    int *A_d, *A1_d, *B_d, *B1_d, *C_d;
    CHECK(cudaMalloc((void**)&A_d, sizeof(int) * sa));
    CHECK(cudaMalloc((void**)&A1_d, sizeof(int) * sa));
    CHECK(cudaMalloc((void**)&B_d, sizeof(int) * sb));
    CHECK(cudaMalloc((void**)&B1_d, sizeof(int) * sb));
    CHECK(cudaMalloc((void**)&C_d, sizeof(int) * sa * sb));

    dim3 blockPerGrid1a((sa - 1)/BLOCKSIZE + 1);
    dim3 threadsPerBlock(BLOCKSIZE);
    dim3 blockPerGrid1b((sb - 1)/BLOCKSIZE + 1);
    dim3 blockPerGrid1c((sb - 1)/BLOCKSIZE + 1, (sa - 1)/BLOCKSIZE + 1);
    dim3 threadsPerBlock2(BLOCKSIZE, BLOCKSIZE);


    CHECK(cudaMemcpy(A_d, A, sizeof(int) * sa, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B, sizeof(int) * sb, cudaMemcpyHostToDevice));
    multKernel<<<blockPerGrid1a, threadsPerBlock>>>(A_d, sa, VALUE, A1_d);
    CHECK_KERNELCALL();
    multKernel<<<blockPerGrid1b, threadsPerBlock>>>(B_d, sb, VALUE, B1_d);
    CHECK_KERNELCALL();
    compareKernel<<<blockPerGrid1c, threadsPerBlock2>>>(A1_d, B1_d, sa, sb, C_d);
    CHECK_KERNELCALL();
    // Or, with the shared memory
    compareKernelShared<<<blockPerGrid1c, threadsPerBlock2>>>(A1_d, B1_d, sa, sb, C_d);
    CHECK_KERNELCALL();

    CHECK(cudaMemcpy(A, A_d, sizeof(int) * sa, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(B, B_d, sizeof(int) * sb, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(C, C_d, sizeof(int) * sb * sa, cudaMemcpyDeviceToHost));
    
    printM(A, 1, sa);
    printM(B, 1, sb);
    printM(C, sa, sb);

    CHECK(cudaFree(A_d));
    CHECK(cudaFree(A1_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(B1_d));
    CHECK(cudaFree(C_d));

    // execute on CPU
    mult(A, sa, VALUE, A1);
    mult(B, sb, VALUE, B1);
    compare(A1, B1, sa, sb, C);

    // print results
    printM(A, 1, sa);
    printM(B, 1, sb);
    printM(C, sa, sb);

    free(A);
    free(B);
    free(A1);
    free(B1);
    free(C);

    return 0;
}
