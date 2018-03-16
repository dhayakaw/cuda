# Chapter 2: CUDA PROGRAMMING MODEL

## Introduction 
### CUDA Programming Structure
- Host: the CPU and its memory
- Device: the GPU and its memory

Kernel: the code that runs on the GPU device

Compiler: nvcc (NVIDIA C Compliler)

A typical processing flow of a CUDA program
1. Copy data from CPU memory to GPU memory
2. Invoke kernels to operate on the data stored in GPU memory
3. Copy data back from GPU memory to CPU memory

### Managing Memory
- `cudaMalloc`: memory allocation
```
cudaError_t cudaMalloc (void** devPtr, size_t size) ```
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

### Organizing Threads
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

## Launching a CUDA Kernel
```
kernel_name <<<grid, block>>> (argument list);
```

## Writing Your Kernel
```
__global__ void kernel_name(argument list);
```
- `__device__`: callable from the device only
- `__host__`: callable from the host only

**CUDA KERNELS ARE FUNCTIONS WITH RESTRICTION**
- Access to device memory only
- Must have `void` return type
- No support for a variable number of arguments
- No support for static variables
- No support for function pointers
- Exhibit an asynchronous behavior

### Verifying Your Kernel
```c
void checkResult(float *hostRef, float *gpuRef, const int N) { 
  double epsilon = 1.0E-8;
  int match = 1;
  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) { 
      match = 0;
      printf("Arrays do not match!\n");
      printf("host %5.2f gpu %5.2f at current %d\n",
          hostRef[i], gpuRef[i], i); 
      break;
    } 
  }
  if (match) printf("Arrays match.\n\n");
  return; 
}
```

### Handling Errors
```c
#define CHECK(call)                                                     \
{                                                                       \
  const cudaError_t error = call;                                       \
  if(error!=cudaSuccess) {                                              \
    printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
    printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \ }                                                                     \
}                                                                       \
```
```c
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
```
```c
kernel_function<<<grid, block>>>(argument list); 
CHECK(cudaDeviceSynchronize());
```

## Timing Your Kernel
### Timing with CPU Timer
- CPU timer: `gettimeofday()`, which returns the number of seconds since the epoch. Include `sys/time.h` header file.
- To measure kernel: `cpuSecond()`:sumArraysOnGPU-timer.cu and `cudaDeviceSynchronize()`

### Timing with nvprof
`nvprof`: a command-line profiling tool
~~~
nvprof [nvprof_args] <application> [application_args]
~~~

## Organizing Parallel Threads
### Index Matrices with Blocks and Threads
2D case
1. Map the thread and block index
```c
ix = threadIdx.x+blockIdx.x*blockDim.x
iy = threadIdy.y+blockIdy.y*blockDim.y
```

2. Map a matrix coordinate to a global memory location/index
```c
idx = iy*nx+ix
```
:`nx*ny` matrix

`printThreadInfo` is used to print out
- Thread index
- Block index
- Matrix coordinate
- Global linear memory offset
- Value of corresponding elements

## Managing Devices
### Using the Runtime API to Query GPU Information
```
cudaGetDeviceProperties()
```
and so on..

### Determining the Best GPU
### Using nvidia-smi to Query GPU Information
```
nvidia-smi -q -i 0
```

### Setting Devices at Runtime
```
CUDA_VISIBL_DEVICES
```
