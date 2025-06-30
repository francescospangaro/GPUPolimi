# Question 1
Explain in few sentences which are the main challenges in 
accelerating histogram computation onto GPU and how they are 
addressed.

## Response
The main challenges are the atomic updates needed because the algorithm consist in a one-to-may function (multiple threads have to increment concurrently the same counters).
The main strategy in order to overcome this challenge is the privatization. In fact we can use the shared memory in order to create "partial results" for each block, and afterwards go to commit these results into the main memory. This strategy is effective since by dividing the results for each block we lower the concurrency between threads (before all threads that were trying to increment the counter of letter "a" for instance were concurrent, now are concurrent only the threads of the SAME block that tries to increment the counter of the letter "a")

# Question 2

In the following snippet of code, how many times will foo() and bar() be 
executed? Motivate the answer.
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
The foo() function will be executed 64 times. One time for each gang spawned by the clause num_gangs(64) which will have 1 worker and 1 vector element.
The bar(i) function will be executed N times, across the different 64 gangs since it has been used the directive loop within the directive parallel

# Question 3
Describe the benefits of the unified memory introduced in the Pascal 
architecture.

## Response
The unified memory introduced in Pascal architecture allow the programmer to use a single address space in order to access both Host and Device memory. This grant to the programmer a much more easier data management in the applicatioj, since the programmer has not to worry about transferring data from Host mem. to Device mem. or viceversa, or about page fault errors which are all handled transparently by the CUDA driver. However this type of addressing cause also performance degradation w.r.t. the case in which the programmer manually handle the data transfers.

# Question 4
Explain which are the benefits of the introduction of a unified shader in 
the Tesla architecture.

## Response
The benefit of the introduction of a unified shader in Tesla architecture is that the single shader processors of the various graphic pipeline's stages aren't anymore the bottleneck of the architecture. In fact with one shader processors for each graphic pipeline's, different type of tasks, for instance vertex-intensive tasks or pixel-intensive tasks would have made the shader processor assigned to the relative pipeline's stage to be the bottleneck of the computation. This problem does not exists anymore with a single shader processor

# Question 5
Which is the efficiency of global load and store operations of the 
following CUDA kernel? Assume to have a block of 32 threads and to 
run the code on a Maxwell architecture where L1 cache has 32-byte 
access width. Motivate the answer.
```
__global__ void vsumKernel(char* a, char* b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    b[i] = a[(i+1)%blockDim.x]; 
}
```

## Response
The index used for the load operations is (global_thread_id+1)%32, meaning that the threads with index from 0 to 30 will access the element from index 1 to 31, and the thread with index 31 will access the index 0. Therefore the efficiency is 100%
From the store point of view the index used is (global_thread_id), meaning that all 32 threads will access the element at their respective index, therefore the efficiency is 100% even in this case.
