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

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
  float *ia = A;
  float *ib = B;
  float *ic = C;

  for(int iy=0; iy<ny; iy++) {
    for(int ix=0; ix<nx; ix++) {
      ic[ix]=ia[ix]+ib[ix];
    }
    ia += nx;
    ib += nx;
    ic += nx;
  }
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny) {
  unsigned int ix = threadIdx.x+blockIdx.x*blockDim.x;
  unsigned int iy = threadIdx.y+blockIdx.y*blockDim.y;
  unsigned int idx = iy*nx+ix;

  if (ix<nx && iy<ny) {
    MatC[idx] = MatA[idx] + MatB[idx];
  }
}

int main (int argc, char **argv) {
  std::cout << argv[0] << " Starting..." << std::endl;

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  std::cout << "Using Device " << dev << ": " << deviceProp.name << std::endl;
  CHECK(cudaSetDevice(dev));

  // set up data size of matrix
  int nx = 1<<12;
  int ny = 1<<12;

  int nxy=nx*ny;
  int nBytes=nxy*sizeof(float);
  std::cout << "Matrix size: " << nx << ", " << ny << std::endl;

  // malloc host memory
  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A    =(float *)malloc(nBytes);
  h_B    =(float *)malloc(nBytes);
  hostRef=(float *)malloc(nBytes);
  gpuRef =(float *)malloc(nBytes);

  // cpuSecond()
  double iStart, iElaps;

  // initial dataset at host site
  iStart=cpuSecond();
  initialData(h_A, nxy);
  initialData(h_B, nxy);
  iElaps=cpuSecond()-iStart;

  memset(hostRef, 0, nBytes);
  memset(gpuRef,  0, nBytes);

  // add vector at host side for result checks
  iStart=cpuSecond();
  sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
  iElaps=cpuSecond()-iStart;

  // malloc device global memory
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, nBytes);
  cudaMalloc((void **)&d_B, nBytes);
  cudaMalloc((void **)&d_C, nBytes);

  // transfer data from host to device
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

  // invoke kernel at host side
  int dimx=32;
  int dimy=32;
  dim3 block(dimx, dimy);
  dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

  iStart=cpuSecond();
  sumMatrixOnGPU2D <<<grid, block>>> (d_A, d_B, d_C, nx, ny);
  cudaDeviceSynchronize();
  iElaps=cpuSecond()-iStart;
  std::cout << "sumMatrixOnGPU2D <<< (" << grid.x << ", " << grid.y << "), ("
    << block.x << ", " << block.y << ") >>>" <<
    "Time elapsed " << iElaps << std::endl;

  // copy kernel result back to host side
  cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

  // check device results
  checkResult(hostRef, gpuRef, nxy); 

  // free device global memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // free host memory
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  cudaDeviceReset();

  return 0;
}

