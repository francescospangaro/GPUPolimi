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

#include <stdio.h>
#include <stdlib.h>

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

#define MAXVAL 100
#define VALUE 10
#define BLOCKDIM 32

void printM(int *M, int numMRows, int numMColumns);
void compare(int *M, int *N, int dm, int dn, int *P);
void mult(int *V, int dim, int fatt, int *P);

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

// kernel function 2: compare each element of M against any element of N
void compare(int *M, int *N, int dm, int dn, int *P)
{
    int i, j;
    for (i = 0; i < dm; i++)
        for (j = 0; j < dn; j++)
            P[i * dn + j] = (M[i] > N[j]);
}

// kernel function 1: vector per scalar multiplication
__global__ void multKernel(int const *const __restrict__ V,
                           int const dim,
                           int const fatt,
                           int *const __restrict__ P)
{
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < dim)
        P[i] = V[i] * fatt;
}

// kernel function 2: compare each element of M against any element of N
__global__ void compareKernel(int const *const __restrict__ M,
                              int const *const __restrict__ N,
                              int const dm,
                              int const dn,
                              int *const __restrict__ P)
{
    // By swapping x and y, we get coalesced read on N and coalesced write on P, but not on M
    int const j = blockDim.x * blockIdx.x + threadIdx.x;
    int const i = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < dm && j < dn)
        P[i * dn + j] = (M[i] > N[j]);
}

__global__ void compareSharedMemKernel(int const *const __restrict__ M,
                                       int const *const __restrict__ N,
                                       int const dm,
                                       int const dn,
                                       int *const __restrict__ P)
{
    // By swapping x and y, we get coalesced write on P
    int const j = blockDim.x * blockIdx.x + threadIdx.x;
    int const i = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ int M_tile[BLOCKDIM];
    __shared__ int N_tile[BLOCKDIM];

    int const i_cp = blockDim.y * blockIdx.y + threadIdx.x; // Use the x thread id to make the copy coalesced
    if (i_cp < dm && threadIdx.y == 0 /* Only 1 thread per each array pos */)
        M_tile[threadIdx.x] = M[i_cp];
    if (j < dn && threadIdx.y == BLOCKDIM - 1 /* Only 1 thread per each array pos */)
        N_tile[threadIdx.x] = N[j];
    __syncthreads();

    if (i < dm && j < dn)
        P[i * dn + j] = (M_tile[threadIdx.y] > N_tile[threadIdx.x]);
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

    // execute on CPU
    mult(A, sa, VALUE, A1);
    mult(B, sb, VALUE, B1);
    compare(A1, B1, sa, sb, C);

    // GPU

    int *A_d, *B_d, *A1_d, *B1_d, *C_d, *C_h;

    C_h = (int *)malloc(sizeof(int) * sa * sb);
    if (!C)
    {
        printf("Error: malloc failed\n");
        free(A);
        free(B);
        free(A1);
        free(B1);
        free(C);
        return 1;
    }

    CHECK(cudaMalloc(&A_d, sizeof(int) * sa));
    CHECK(cudaMalloc(&A1_d, sizeof(int) * sa));
    CHECK(cudaMalloc(&B_d, sizeof(int) * sb));
    CHECK(cudaMalloc(&B1_d, sizeof(int) * sb));
    CHECK(cudaMalloc(&C_d, sizeof(int) * sa * sb));

    CHECK(cudaMemcpy(A_d, A, sizeof(int) * sa, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B, sizeof(int) * sb, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCKDIM);
    multKernel<<<(sa + BLOCKDIM - 1) / BLOCKDIM, threadsPerBlock>>>(A_d, sa, VALUE, A1_d);
    multKernel<<<(sb + BLOCKDIM - 1) / BLOCKDIM, threadsPerBlock>>>(B_d, sb, VALUE, B1_d);

    threadsPerBlock = dim3(BLOCKDIM, BLOCKDIM);
    dim3 numOfBlocks((sb + BLOCKDIM - 1) / BLOCKDIM, (sa + BLOCKDIM - 1) / BLOCKDIM);
    compareKernel<<<numOfBlocks, threadsPerBlock>>>(A1_d, B1_d, sa, sb, C_d);

    CHECK(cudaMemcpy(C_h, C_d, sizeof(int) * sa * sb, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(A_d));
    CHECK(cudaFree(A1_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(B1_d));
    CHECK(cudaFree(C_d));

    // print results
    printM(A, 1, sa);
    printM(B, 1, sb);
    printM(C_h, sa, sb);

    free(C_h);
    free(A);
    free(B);
    free(A1);
    free(B1);
    free(C);

    return 0;
}
