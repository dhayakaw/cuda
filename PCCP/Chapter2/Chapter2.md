# Chapter 2: CUDA PROGRAMMING MODEL

## CUDA Programming Structure
- Host: the CPU and its memory
- Device: the GPU and its memory

Kernel: the code that runs on the GPU device

Compiler: nvcc (NVIDIA C Compliler)

A typical processing flow of a CUDA program
1. Copy data from CPU memory to GPU memory
2. Invoke kernels to operate on the data stored in GPU memory
3. Copy data back from GPU memory to CPU memory

## Managing Memory
- `cudaMalloc`: memory allocation
```
cudaError_t cudaMalloc (void** devPtr, size_t size)
```

- `cudaMemcpy`: transfer data btw the host and device
```
cudaError_t cudaMemcpy (void* dst, const void* src, size_t count, cudaMemcpyKind kind)
```
copy src (source memory area) to dst (destination memory area) with kind ,where kind takes one of the following type:
  - `cudaMemcpyHostToHost`
  - `cudaMemcpyHostToDevice`
  - `cudaMemcpyDeviceToHost`
  - `cudaMemcpyDeviceToDevice`

return `cudaSuccess` if GPU memory is successfully allocated. Otherwise, it returns `cudaErrorMemoryAllocation`
```
char* cudaGetErrorString(cudaError_t error)
```
converts error to a human-redable error message

- `cudaMemset`
- `cudaFree`

LISTING: summing two arrays

## Organizing Threads
All threads spawned by a single kernel launch are collectively called a grid. A grid is made up of many thread blocks.
~~~
threads << blocks <<< a grid
- blockIdx (block index within a grid)
- threadIdx (thread index within a block)
~~~
`uint3`: the type of the coordinate variable -> 3 components (x, y and z) are accessible
```
- blockIdx.x
- blockIdx.y
- blockIdx.z
- threadIdx.x
- threadIdx.y
- threadIdx.z
```

`dim3`: threedimensions
```
- blockDim.x
- blockDim.y
- blockDim.z
``` 

**Usually, a grid is organized as a 2D array of blocks and a block is organized as a 3D array of threads.**

**You define variables for grid and block on the host before launching a kernel, and a ccess them there with the x, y and z fields of the vector structure from the host side.**


