# Professional CUDA C Programming

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
- cudaMalloc

memory allocation
```
cudaError_t cudaMalloc (void** devPtr, size_t size)
```

- cudaMemcpy
transfer data btw the host and device
```
cudaError_t cudaMemcpy (void* dst, const void* src, size_t count, cudaMemcpyKind kind)
```
copy src (source memory area) to dst (destination memory area) with kind ,where kind takes one of the following type:
  - cudaMemcpyHostToHost
  - cudaMemcpyHostToDevice
  - cudaMemcpyDeviceToHost
  - cudaMemcpyDeviceToDevice

return `cudaSuccess` if GPU memory is successfully allocated. Otherwise, it returns `cudaErrorMemoryAllocation`

- cudaMemset

- cudaFree
