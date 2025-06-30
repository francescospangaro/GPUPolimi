# Question 1 

Consider  the  following  histogram  kernel  exploiting  privatization,  shared  memory,  and  thread  coarsening  using 
contiguous  partitions.  Let’s  assume  that  the  kernel  processes  an  input  with  524,288  elements  to  produce  a 
histogram with 5 bins and that it is configured with 1024 threads per block; what is the maximum number of atomic 
operations that the kernel may perform on global memory? Motivate the answer. 

```
#define CHAR_PER_BIN  6 
#define ALPHABET_SIZE 26 
#define BIN_NUM ((ALPHABET_SIZE - 1) / CHAR_PER_BIN + 1) /* equal to 5 */ 
#define FIRST_CHAR 'a' 
 
__global__ void histogram_kernel(char *data, int *histogram, int length) { 
  int tid    = blockIdx.x * blockDim.x + threadIdx.x; 
  int stride = blockDim.x * gridDim.x; 
  __shared__ int histo_s[BIN_NUM]; 
  for (int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x)  
    histo_s[binIdx] = 0; 
  __syncthreads(); 
 
  for (int i = tid; i < length; i += stride) { 
    int alphabet_position = data[i] - FIRST_CHAR; 
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE) 
      atomicAdd(&(histo_s[alphabet_position / CHAR_PER_BIN]), 1); 
  } 
  __syncthreads(); 
 
  for (int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) { 
    int binValue = histo_s[binIdx]; 
    if (binValue > 0) { 
      atomicAdd(&(histogram[binIdx]), binValue); 
    } 
  } 
}
```

## Response:

The maximum number of operations thath the kernel my perform on global memory is:
524.288/1024 = 512, which is also the number of blocks.
In fact through the privatization technique, by using the shared memory, we are able through
create partial results associated with each block spawned by the kernel config.
As last thing we need to commit this partial result in one final global result
resulting in 512*5 (due to the necessity to commit every bin value to the
global memory through atomicAdds) operations to the global memory.

N.B: The number of blocks (512) is just a guess, since from the code we have no
insight on the number of blocks used in the kernel function. However 512 is the
minum number of blocks needed in order to assign to each thread a value of the
"data" parameter.



# Question 2 

Describe the benefits of the unified memory introduced in the Pascal architecture. 
 
## Response:

The unified memory is a tehcnology introduced in Pascal architecture in order to
make the memory management easier from the point of view of the user (programmer).
In fact through the unified memory the architecture handle the transmission in a
transparent way of the memory pages, and therefore data, from the host memory to the
device memory. Moreover the unified memory handle also the eventual page faults in
an automatic way. This is all based on an unified virtual memory addressing between
the host and device memory.
This strategy simplify hugely the memory usage from the point of view of the programmer
however it also degrades the overall kernels performances.
The unified memory can be used through the CUDA call: cudaMallocManaged(...) which
let the programmer allocate space as in a unified memory.


# Question 3 
Explain the benefits of introducing a unified shader core in Tesla architecture. 

## Response
The unified shader core was introduced in Tesla architecture due to bottleneck problems
of the previous GPUs.
In fact in the previous GPUs there were several shared cores, one for each stage of 
the graphic pipeline. However different graphic pipeline tasks can have very different
characteristics, for instance we can have very vertex-intensive tasks or others tasks
can be pixel-intensive ecc. For this reason different shader core could have been
the bottleneck of the graphic pipeline based on the type of the task.
By introducing a single (and bigger) unified shader processor, we can handle better
these type of different tasks by executing more shader program concurrently even of
different stages.

# Question 4 
Let’s assume to run the following kernel on a Maxwell (or more recent) architecture and to size the grid with a single 
block  of  32  threads;  which  is  the  efficiency  of  global  load  and  store  operations  of  the  following  CUDA  kernel? 
Motivate the answer. 
 
 ```
__global__ void foo(char* a, char* b){ 
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  b[i+1] = a[(i+1)%blockDim.x];  
}
```
## Response
For what concern the load operations (READ), the memory is accessed in a aligned coalesced 
way. In fact as index we are using (i+1)%blockDim.x = (#global_id_thread + 1)% 32.
This means that the threads with id from 0 to 30, will access the vector a's values
from index 1 to index 31, the thread with index 31 will access the value at index 0.
However this is not a problem since all 32 threads are accessing contiguous
memory address therefore they can all be handled by using a single transaction.
Therefore the overall memory efficiency is 100%.

From the point of view of the store operations (WRITE), the same reasoning can be
applied, however differently from the previous case, we have that the indexes spans
from 1 to 32 resulting in accessing 2 32-byte "memory region" and therefore resulting
in 2 transaction.
The overall memory efficiency is 50% (We loaded 64 memory addresses, and accessed only 32)



# Question 5 
In the following snippet of code using OpenACC pragmas, how many times will foo() and bar() be executed? 
Motivate the answer. 

```
#pragma acc parallel num_gangs(64) 
{ 
  foo();  
  #pragma acc loop gang 
  for (int i=0; i<n; i++) { 
    bar(i); 
  } 
}
```

## Response
foo() function will be executed 64, since through the directive parallel and clause
num_gangs(64) we spawn 64 gangs, each one with 1 worker and 1 vector element that
will redundantly execute all the same code.
bar() function will be instead executed N times, since we parallelize the for loop
with the directive loop that will "distribute" the N iterations in threads across
the 64 gangs (resulting therefore in executing bar() only N times and NOT N * 64 times).
