# Question 1 
Explain which are the three features of the graphics pipeline that have been parallelized in the GPU architecture.

# Response
In the GPU architecture have been parallelized 3 features of the graphic pipeline:
1. Vertex stage: Which include two steps
    - Vertex generation: During this step the verteces (point in 3D) are generated
    - Vertex processing: During this step there are several trasformation regarding the verteces and matrix multiplication. It's usually also the phase in which the memory communication is higher due to the reading of the textures.
2. Primitive stage:
    - Primitive generation: Verteces are taken and transformed in primitives (polygons with basic shapes such as triangle)
    - Primitive processing: The generated primitives are subjected to several transformation such as: clipping, view point transformation, perspective division,...)
3. Fragment stage:
    - Fragments generation (Rasterization): The different primitives are converted into fragments that describe to which pixels the primitives will overlap. An initial coloring of the scene is started.
    - Fragment processing: The coloring is completed even taking care of scene characteristics such as the lighting and the objects' interaction.

Question 2 
Draw the Gantt chart of the execution of the various functions in the following three cases. 
 
 ```
(a) Assume that foo execution is shorter than cpuFoo one. 
foo<<<blocks, threads>>>();  
cpuFoo(); 
foo<<<blocks, threads>>>();  
cudaEventRecord(event1); 
cudaEventSynchronize(event1); 
cpuFoo(); 
 ```
 ```
(b) Assume that foo execution is longer than goo one. 
cudaStreamCreate(&stream1);  
cudaStreamCreate(&stream2);  
foo<<<blocks, threads, 0, stream1>>>();  
cudaEventRecord(event1, stream1); 
goo<<<blocks, threads, 0, stream2>>>();  
cudaStreamWaitEvent(stream2, event1); 
foo<<<blocks, threads, 0, stream2>>>();  
 ```
 ```
(c) Assume that foo execution is longer than goo one. 
cudaStreamCreate(&stream1);  
cudaStreamCreate(&stream2);  
foo<<<blocks, threads, 0, stream1>>>();  
goo<<<blocks, threads, 0, stream2>>>();  
foo<<<blocks, threads>>>();  
foo<<<blocks, threads, 0, stream2>>>();
```

## Response

# Question 3 
Briefly  describe  CUDA  memory  model;  for  each  component  specify  name,  type  of  usage,  type  of  access 
(read/write or read only) and scope.

## Response
|Name|Usage|Access|Scope|
|----|-----|------|-----|
|Registers|Used to store automatic (primitive) variables such as int, double, ...|Device: Read/Write| Single thread|
|Local mem.|Used to store "large" automatic variables such as arrays of integer, structs, ...|Device: Read/Write|Single thread|
|Global mem.|Used move data from the Host to the Device and viceversa, variables can be declared statically or dynamically|Device: Read/Write <br> Host: Read/Write|Entire application|
|Shared mem.|Used to allow threads from the same block to cooperate and exchange data/informations between them|Device: Read/Write|Single block|
|Constant mem.|Used to store constant values, since can handle their readings much more efficiently that global memory for instance|Device: Read-only <br> Host: Read/Write|Entire application|
|Texture mem.|Used to store matrices of data (aka textures) since it can handle them very efficiently|Device: Read-only <br> Host: Read/Write|Entire application|

# Question 4 
Describe the main strategies to accelerate the Smith-Waterman algorithm in CUDA.

## Response
>[!NOTE]
>This algorithm has not been covered during our lessons

# Question 5 
In the following snippet of code, how many times will foo() and goo() be executed? Motivate the answer

```
#pragma acc parallel num_gangs(16) 
{ 
  #pragma acc loop gang 
  for (int i=0; i<32; i++) { 
    goo(i); 
  } 
  foo();  
}
```

## Response
The foo() function will be executed 16 times, since with the clause num_gangs(16) associated to the directive parallel, we spawn 16 gangs each with 1 worker and 1 vector element, that will execute the code redundantly.
For what concern the goo() function insted, the functio will be executed 32 times, since with the loop directive openACC will parallelize the 32 for loop's iterations by distributing them across the gangs spawned precedently.
