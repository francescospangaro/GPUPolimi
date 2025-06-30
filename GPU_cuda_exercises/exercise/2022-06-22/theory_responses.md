# Question 1 
Describe the benefits of warp interleaving in NVIDIA GPU and how it is managed. 

## Response
Warp interleaving is a technique that can be used by NVIDIA GPU in order to hide memory accesses latency or stalls. Each warp can be classified as:
1. Selected: If the warp is currently running
2. Stalled: If the warp is not ready to be executed
3. Eligible: If the warp is ready to be exectued (warp scheduler will choose between these type of warps)

The warp interleaving aim is mainly try to hide latencies and achieve the maximum throughput possible

# Question 2 
What will the following program fragment print on the screen? How many threads are created? 
```
__global__ void foo(int iSize, int iDepth) {  
  int tid = threadIdx.x; 
 
  if (iSize > 1) { 
    int nthreads = iSize/2;  
    if(tid == 0 && nthreads > 0){ 
      foo<<<1, nthreads>>>(nthreads, iDepth+1);   
      cudaDeviceSynchronize(); 
    } 
    __syncthreads(); 
  } 
  printf("Recursion=%d: Hi from thread %d block %d\n", iDepth, tid, blockIdx.x);  
} 
int main(){ 
  /*...*/ 
  int iSize = 4; 
  foo<<<1, iSize>>>(iSize, 0); 
  /*...*/ 
} 
```

## Response
"Recursion=2: Hi from thread 0 block 0"
"Recursion=1: Hi from thread 0 block 0"
"Recursion=1: Hi from thread 1 block 0"
"Recursion=0: Hi from thread 0 block 0"
"Recursion=0: Hi from thread 1 block 0"
"Recursion=0: Hi from thread 2 block 0"
"Recursion=0: Hi from thread 3 block 0"

Total threads created: 7

# Question 3 
Briefly  describe  CUDA  memory  model;  for  each  component  specify  name,  type  of  usage,  type  of  access 
(read/write or read only) and scope.

## Response

|Name|Usage|Access|Scope|
|----|-----|------|-----|
|Registers|Used to store automatic (primitive) variables such as int, float, ...|Device: Read/Write|Single thread|
|Local mem.|Used to store automatic variables of big size (arrays, structs for instance...)|Device: Read/Write|Single thread|
|Global mem.|Used by programmer to store every type of vars that they want through a static or dynamic declaration|Device: Read/Write <br> Host: Read/Write|Entire application|
|Shared mem.|Used by the programmer to store data that typically get accessed a lot by threads, since accessing it it's much faster than accessing global mem. Data can be stored with a static or dynamic declaration|Device: Read/Write|Single block|
|Constant mem.|Used to store constant values that must be accessed by multiple threads|Device: Read-only <br> Host: Read/Write|Entire application|
|Texture mem.|Used typically to store 2D arrays (aka textures), since it can handle them very efficiently|Device: Read-only <br> Host: Read/Write|Entire application|

# Question 4 
Comment  the  benefits  of  the  unified  coherent  memory  architecture  implemented  in  AMD  heterogeneous 
systems w.r.t. the memory organization of a traditional architecture having a discrete GPU.

## Response

>[!NOTE]
>Argument not covered during the course

# Question 5 
What is the key optimization to speed up the convolution process? 

## Response
The key optimization to speed up the convolution process is the constant memory optimization. In fact since the filter with which each input matrix value must be multiplied is always the same, we can use the constant memory to store in it such filter. This hugely speedup the computation since each thread now has to access the constant memory instead of the global memory to access filter's values (constant memory has an access latency very small compared to the global memory one)

