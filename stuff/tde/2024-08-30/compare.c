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

#define MAXVAL 100
#define VALUE 10

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
