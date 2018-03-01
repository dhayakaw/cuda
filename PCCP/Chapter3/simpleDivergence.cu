#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void warmingup(float *c) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  float a, b;
  a=b=0.0f;
  if (tid%2==0){
    a=100.0f;
  } else {
    b=200.0f;
  }
  c[tid] = a+b;
}

__global__ void mathKernel1(float *c) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  float a, b;
  a=b=0.0f;
  if (tid%2==0){
    a=100.0f;
  } else {
    b=200.0f;
  }
  c[tid] = a+b;
}

__global__ void mathKernel2(float *c) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  float a, b;
  a=b=0.0f;
  if (tid%2==0){
    a=100.0f;
  } else {
    b=200.0f;
  }
  c[tid] = a+b;
}
__global__ void mathKernel3(float *c) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  float a, b;
  a=b=0.0f;
  bool ipred = (tid&2==0);
  if (ipred){
    a=100.0f;
  } else {
    b=200.0f;
  }
  c[tid] = a+b;
}

int main(int argc, char **argv) {
  // setup device
  int dev=0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  std::cout << argv[0] << " using device " << dev << ": " << deviceProp.name << std::endl;

  // set up data size
  int size = 64;
  int blocksize = 64;
  if(argc>1) blocksize = atoi(argv[1]);
  if(argc>2) blocksize = atoi(argv[2]);
  std::cout << "Data size " << size << std::endl;

  // set up execution configuration
  dim3 block (blocksize, 1);
  dim3 grid  ((size+block.x-1)/block.x, 1);
  std::cout << "Execution configure (block " << block.x << " grid " << grid.x << ")" << std::endl;

  // allocate gpu memory
  float *d_C;
  size_t nBytes = size*sizeof(float);
  cudaMalloc((float**)&d_C, nBytes);

  // run a warmup kernel to remove overhead
  cudaDeviceSynchronize();
  auto iStart = std::chrono::system_clock::now();
  warmingup<<<grid, block>>> (d_C);
  cudaDeviceSynchronize();
  auto iElaps = std::chrono::system_clock::now() - iStart;
  auto nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(iElaps).count();
  std::cout << "warmup <<< " << grid.x << " " << block.x << " >>> elapsed " << nsec << std::endl;

  // run kernel 1
  iStart = std::chrono::system_clock::now();
  mathKernel1<<<grid, block>>> (d_C);
  cudaDeviceSynchronize();
  iElaps = std::chrono::system_clock::now() - iStart;
  nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(iElaps).count();
  std::cout << "kernel1 <<< " << grid.x << " " << block.x << " >>> elapsed " << nsec << std::endl;
  // run kernel 2
  iStart = std::chrono::system_clock::now();
  mathKernel2<<<grid, block>>> (d_C);
  cudaDeviceSynchronize();
  iElaps = std::chrono::system_clock::now() - iStart;
  nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(iElaps).count();
  std::cout << "kernel2 <<< " << grid.x << " " << block.x << " >>> elapsed " << nsec << std::endl;
  // run kernel 3
  iStart = std::chrono::system_clock::now();
  mathKernel3<<<grid, block>>> (d_C);
  cudaDeviceSynchronize();
  iElaps = std::chrono::system_clock::now() - iStart;
  nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(iElaps).count();
  std::cout << "kernel3 <<< " << grid.x << " " << block.x << " >>> elapsed " << nsec << std::endl;

  // free gpu memory and reset device
  cudaFree(d_C);
  cudaDeviceReset();
  return EXIT_SUCCESS;
}
