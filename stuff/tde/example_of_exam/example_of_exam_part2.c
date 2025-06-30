/*
* The kernel function to accelerate receives in input a vector of positive integers, called A, 
* together with its size, and a second empty vector of integers, B, of the same size. 
* For each element i in A, the function saves in B[i] the value 1 if A[i] is greater than all the 
* neighbor values with an index between (i-DIST) and (i+DIST), bounds included and if they exist; 
* 0 otherwise. DIST is a constant value defined with a macro.
* The main function is a dummy program that receives as an argument the vector size, instantiates and 
* populates randomly A, invokes the above function, and shows results.
*/

#include <stdio.h>
#include <stdlib.h>

#define MAXVAL 100
#define DIST 10

void printV(int *V, int num);
void compute(int *V, int *R, int num);

//display a vector of numbers on the screen
void printV(int *V, int num) {
  int i;
  for(i=0; i<num; i++)
      printf("%3d(%d) ", V[i], i);
  printf("\n");    
}

//kernel function: identify peaks in the vector
void compute(int *V, int *R, int num) {
  int i, j, ok;
  for(i=0; i<num; i++){
    for(j=-DIST, ok=1; j<=DIST; j++){
      if(i+j>=0 && i+j<num && j!=0 && V[i]<=V[i+j])
        ok=0;
    }
    R[i] = ok;
  }
}


int main(int argc, char **argv) {
  int *A;
  int *B;
  int dim;
  int i;

  //read arguments
  if(argc!=2){
    printf("Please specify sizes of the input vector\n");
    return 0;
  }
  dim=atoi(argv[1]);
  
  //allocate memory for the three vectors
  A = (int*) malloc(sizeof(int) * dim);
  if(!A){
    printf("Error: malloc failed\n");
    return 1;
  }
  B = (int*) malloc(sizeof(int) * dim);
  if(!B){
    printf("Error: malloc failed\n");
    return 1;
  }

  //initialize input vectors
  srand(0);
  for(i=0; i<dim; i++)
    A[i] = rand()%MAXVAL +1;
  
  //execute on CPU  
  compute(A, B, dim);
 
  //print results
  printV(A, dim);
  printV(B, dim);
  
  free(A);
  free(B);
  
  return 0;
}


