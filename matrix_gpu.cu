#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define MATRIX_SIZE 1024
#define BLOCK_SIZE 16

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC);

int main(int argc, char** argv){
unsigned int matrixSize = sizeof(unsigned int) * MATRIX_SIZE * MATRIX_SIZE;

  int* hMatrixA;
  int* hMatrixB;
  int* hMatrixC;
  hMatrixA = (int*)malloc(matrixSize);
  hMatrixB = (int*)malloc(matrixSize);

  unsigned int col_idx, row_idx;
  for (col_idx = 0; col_idx < MATRIX_SIZE; col_idx++){
      for (row_idx = 0; row_idx < MATRIX_SIZE; row_idx++){
          hMatrixA[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024*1024);
          hMatrixB[col_idx * MATRIX_SIZE + row_idx] = rand() % (1024*1024);
      }
  }

  int* dMatrixA;
  int* dMatrixB;
  int* dMatrixC;

  checkCudaErrors(cudaMalloc((void**)&dMatrixA, matrixSize));
  checkCudaErrors(cudaMemcpy(dMatrixA, hMatrixA, matrixSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void**)&dMatrixB, matrixSize));
  checkCudaErrors(cudaMemcpy(dMatrixB, hMatrixB, matrixSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void**)&dMatrixC, matrixSize));

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(MATRIX_SIZE/BLOCK_SIZE, MATRIX_SIZE/BLOCK_SIZE);

  cudaEvent_t start;
  cudaEvent_t stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start, NULL)); //start

  matrixMul<<<grid, block>>>(dMatrixA, dMatrixB, dMatrixC);
  cudaThreadSynchronize();

  hMatrixC = (int*)malloc(matrixSize);
  checkCudaErrors(cudaMemcpy(hMatrixC, dMatrixC, matrixSize, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaEventRecord(stop, NULL));
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  printf("Processing time: %f (msec)\n", msecTotal);

  free(hMatrixA);
  free(hMatrixB);
  free(hMatrixC);
  checkCudaErrors(cudaFree(dMatrixA));
  checkCudaErrors(cudaFree(dMatrixB));
  checkCudaErrors(cudaFree(dMatrixC));

  cudaThreadExit();
  exit(1);
}

__global__ void
matrixMul(int* inMatrixA, int* inMatrixB, int* inMatrixC){
  unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int scan_idx;
  unsigned int target = 0;

 for (scan_idx = 0; scan_idx < MATRIX_SIZE; scan_idx++) {
   target +=inMatrixA[col_idx * MATRIX_SIZE + scan_idx] * inMatrixB[scan_idx * MATRIX_SIZE + row_idx];
   __syncthreads();
 }
 inMatrixC[col_idx * MATRIX_SIZE + row_idx] = target;
}
