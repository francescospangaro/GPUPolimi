# Question 1 
Explain the peculiarities of the SIMT approach adopted in NVIDIA GPUs w.r.t. the classical vector processors.

## Response
The SIMT(Single Instruction Multiple Thread) is an approach adpoted by NVIDIA as evolution of the SIMD approach. With the SIMT approach in fact a NVIDIA GPU can execute the same instructions on several different threads working on different data element. The main difference is that these threads can access their own register perform load and store from divergent addresses, ecc...

# Question 2 
Complete  the  following  OpenACC  pragmas  to  optimize  data  transfer  between  the  host  and  the  device.  In 
particular, 1) specify between brackets () the appropriate list of arrays (index range for each array is always 
[0:N]), and 2) delete unnecessary clauses. Motivate your answer. 

```
void foo(int *A, int *B, int *D){ 
  int C[N]; 
#pragma acc data copy() copyin() copyout() create()   
{ 
  #pragma acc parallel loop copy() copyin() copyout() create() 
  for(int i = 0; i < N; i++) 
    C[i] = A[i] + B[i]; 
  #pragma acc parallel loop copy() copyin() copyout() create()
  for(int i = 0; i < N; i++) 
    D[i] = C[i] + A[i]; 
} 
} 
```

## Response
<pre>
void foo(int *A, int *B, int *D){ 
  int C[N]; 
#pragma acc data <s>copy()</s> copyin(A[0:N]) <s>copyout(     )</s> create(C[0:N])   
{ 
  #pragma acc parallel loop <s>copy()</s> copyin(B[0:N]) <s>copyout()</s> <s>create()</s> 
  for(int i = 0; i < N; i++) 
    C[i] = A[i] + B[i]; 
  #pragma acc parallel loop <s>copy()</s> <s>copyin()</s> copyout(D[0:N]) <s>create()</s>
  for(int i = 0; i < N; i++) 
    D[i] = C[i] + A[i]; 
} 
} 
</pre>

We can use the first copyin in order to move array A from host mem. to device mem. and use it for the 2 inner for loops.
We can use the first create() since array C is only used device side, therefore we need only to allocate space on the device memory, and it's used by both inner for loops.
We can use the second copyin() in order to move B form host memory to device memory since only the first for loop use it.
We can use the 3rd copyout() to move the D array from device memory to host memory since only the second loop use it.


# Question 3 
Explain what pinned memory is and why it is necessary when using asynchronous data transfer. 

## Response
Pinned memory is a page-locked memory region that is allocated automatically by the CUDA driver or manually by the programmer using cudaMallocHost() in order to enable asynchronous data transfers. In fact the only way for the device to access in a safe way the host memory, is in the case the memory is page-locked, therefore when an asynchronous data transfer is invoked the CUDA driver pin some host memory, move the data to be transferred to this region of the memory and then proceeds with the asynchronous data transfer from host memory to device memory
 
# Question 4 
Which is the efficiency of global load and store operations of the following CUDA kernel? Assume to have a block of 
32 threads and to run the application on a Maxwell architecture where L1 cache has 32-byte access width. Motivate 
your answer. 
 
```
__global__ void vsumKernel(char* a, char* b){ 
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  b[(i+2)%blockDim.x] = a[(i*2)%blockDim.x];  
}
```

## Response
From the load point of view, the used index is (global_thread_id*2)%32, therefore the first 16 threads (from id 0 to 15) will access the even positions from 0 to 30, and the last 16 threads will do the same. Therefore the efficiency is 50%.

From the store point of view, the used index is (global_thread_id+2)%32, therefore the threads will access indexes shifted by 2 w.r.t their ids: 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 0 1.
Therefore the efficiency is 100%

# Question 5 
Explain which is the bottleneck in the performance of the basic histogram algorithm and how it can be solved.

## Response
The bottleneck in the performance of the histogram algorithm is the moltitude of atomic adds that must be performed by the several threads to the same memory address (which are very small compared to the number of threads). The main strategy that can be used in order to solve this bottleneck is the privatization: Using shared memory we can create several partial histograms, each one accessed only by the threads within the same block. In this way we reduce heavily the concurrency between threads and we can also exploit the lower latency offered by the shared memory w.r.t the global memory. 
Finally we need to commit all the partial results in a unique histogram stored in the global memory.
