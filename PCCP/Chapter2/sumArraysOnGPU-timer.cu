#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#define CHECK(call)                                                     \
{                                                                       \
  const cudaError_t error = call;                                       \
  if(error!=cudaSuccess) {                                              \
    printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
    printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
  }                                                                     \
}                                                                       \

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = true;
  for(int i=0; i<N; i++){
    if (abs(hostRef[i]-gpuRef[i])>epsilon){
        match = false;
        std::cout << "Arrays do not match!" << std::endl;
        printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i],i);
        break;
    }
  }
  if (match) std::cout << "Arrays match. " << std::endl;
}

void initialData(float *ip, int size) {
  // generate different seed for random number
  time_t t;
  srand((unsigned int) time(&t));

  for(int i=0; i<size; i++) {
    ip[i] = (float)(rand()&0xFF)/10.f;
  }
}

void sumArraysOnHost(float *A, float *B, float *hostRef, const int N) {
  for (int idx=0; idx<N; idx++) {
    hostRef[idx] = A[idx]+B[idx];
  }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<N) C[i]=A[i]+B[i];
}


int main (int argc, char **argv) {
  std::cout << argv[0] << " Starting..." << std::endl;

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  std::cout << "Using Device " << dev << ": " << deviceProp.name << std::endl;
  CHECK(cudaSetDevice(dev));

  // set up data size of vectors
  int nElem=1<<24;
  std::cout << "Vector size: " << nElem << std::endl;

  // malloc host memory
  size_t nBytes = nElem*sizeof(float);

  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A    =(float *)malloc(nBytes);
  h_B    =(float *)malloc(nBytes);
  hostRef=(float *)malloc(nBytes);
  gpuRef =(float *)malloc(nBytes);

  // cpuSecond()
  double iStart, iElaps;

  // initial dataset at host site
  iStart=cpuSecond();
  initialData(h_A, nElem);
  initialData(h_B, nElem);
  iElaps=cpuSecond()-iStart;

  memset(hostRef, 0, nBytes);
  memset(gpuRef,  0, nBytes);

  // add vector at host side for result checks
  iStart=cpuSecond();
  sumArraysOnHost(h_A, h_B, hostRef, nElem);
  iElaps=cpuSecond()-iStart;

  // malloc device global memory
  float *d_A, *d_B, *d_C;
  cudaMalloc((float**)&d_A, nBytes);
  cudaMalloc((float**)&d_B, nBytes);
  cudaMalloc((float**)&d_C, nBytes);

  // transfer data from host to device
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

  // invoke kernel at host side
  int iLen=1024;
  // int iLen=2048; // Arrays did not match with GTX745 because the maximum number of threads per block is 1024
  dim3 block(iLen);
  dim3 grid ((nElem+block.x-1)/block.x);

  iStart=cpuSecond();
  sumArraysOnGPU <<<grid, block>>> (d_A, d_B, d_C, nElem);
  cudaDeviceSynchronize();
  iElaps=cpuSecond()-iStart;
  std::cout << "Execution configuration <<< " << grid.x << ", " << block.x << ">>>" 
    "Time elapsed " << iElaps << std::endl;

  // copy kernel result back to host side
  cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

  // check device results
  checkResult(hostRef, gpuRef, nElem); 

  // free device global memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // free host memory
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  return 0;
}

