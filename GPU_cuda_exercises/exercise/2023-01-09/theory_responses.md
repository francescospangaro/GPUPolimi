# Question 1 
Explain what a tensor core introduced in the Volta architecture is.

## Response
A tensor core is a computing core which is specialized in dealing with data type in the format of "tensors" which basically are N-dimensional data vectors. It was introduced in the Volta architecture mainly in ordeer to accelerate deep neural network tasks. It is particularly used
for matrix multiplication and accumulation.

# Question 2 
Which is the efficiency of global load and store operations of the following CUDA kernel? Assume to have a block of 
32  threads  and  run  the  code  on  a  Maxwell  architecture  where  L1  cache  has  32-byte  access  width.  Motivate  the 
answer. 

```
__global__ void foo(char* a, char* b){ 
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  b[(i+1)%blockDim.x] = a[(i*blockDim.x)%blockDim.x];  
}
```

## Response
From the load point of view, we are using as index: (global_id_thread*32)%32
which will always access the index 0 of the "a" array. In fact by multiplying every thread
index with 32 we will always obtain a multiple of 32 (obviously) which will result always in a 0 rest division using %32.
Therefore the efficiency is 1/32, 3,125% since we are loading 32 elements but accessing only one of them.

From the store point of view the access pattern is 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0
Which means that the efficiency is 100% since we are writing every memory address accessed.

# Question 3 
Briefly explain the main advantages and drawbacks of OpenACC w.r.t. CUDA.

## Response
OpenACC has several main characteristics:
1. Advantages: 
    - __High portability__: OpenACC guarantees high portability of the code since it can be tageted to different HW accelerators just by recompiling it every time.
    - __High programmability__: OpenACC guarantess also an high programmability since it makes programming parallel code much more easier than (for instance) CUDA, through annotations and directives.
1. Disadvantages: 
    - __Lower performances__: OpenACC is very flexible but we have to trade off this flexibility with the performance aspect, in fact OpenACC is not able to accelerate functions as fast as CUDA can (for instance). 

# Question 4 
Describe the bottleneck in the CUDA-based acceleration of the Monte Carlo simulation algorithm and which 
strategy can be adopted to solve it. 

## Response

> [!NOTE]
> Monte Carlo simulation algorithm has not been covered in the execise sessions

# Question 5 
What will the following program fragment print on the screen? How many threads are created? 

```
__global__ void foo(int size, int depth) {  
  if (depth > 0) { 
    if(threadIdx.x == 0){ 
      foo<<<1, size>>>(size, depth-1);   
      cudaDeviceSynchronize(); 
    } 
    __syncthreads(); 
  } 
  printf("Depth: %d thread: %d block: %d\n", depth, threadIdx.x, blockIdx.x);  
} 
 
int main(){ 
  /*...*/ 
  int size = 4, depth = 2; 
  foo<<<1, size>>>(size, depth); 
  /*...*/ 
}
```

## Response
It will print:
"Depth: 0 thread: 0 block: 0"
"Depth: 0 thread: 1 block: 0"
"Depth: 0 thread: 2 block: 0"
"Depth: 0 thread: 3 block: 0"
"Depth: 1 thread: 0 block: 0"
"Depth: 1 thread: 1 block: 0"
"Depth: 1 thread: 2 block: 0"
"Depth: 1 thread: 3 block: 0"
"Depth: 2 thread: 0 block: 0"
"Depth: 2 thread: 1 block: 0"
"Depth: 2 thread: 2 block: 0"
"Depth: 2 thread: 3 block: 0"

12 threads in total are created.
N.B: The printf at the same depth could appear in different order based on the order in which the different threads complete their execution