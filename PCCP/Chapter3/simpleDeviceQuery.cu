#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

int main(int argc, char*argv[]) {
  int iDev=0;
  cudaDeviceProp iProp;
  cudaGetDeviceProperties(&iProp, iDev);

  cout << "Device " << iDev << ": " << iProp.name << endl;
  cout << "Number of multiprocessors: " << iProp.multiProcessorCount << endl;
  cout << "Total amount of constant memory: " << iProp.totalConstMem/1024.0 << endl;
  cout << "Total number of registers available per block: " << iProp.sharedMemPerBlock << endl;
  cout << "Warp size: " << iProp.warpSize << endl;
  cout << "Maximum number of threads per block: " << iProp.maxThreadsPerBlock << endl;
  cout << "Maximum number of threads per multiprocessor: " << iProp.maxThreadsPerMultiProcessor << endl;
  cout << "Maximum number of warps per multiprocessor: " << iProp.maxThreadsPerMultiProcessor/32 << endl;
  return EXIT_SUCCESS;
}
