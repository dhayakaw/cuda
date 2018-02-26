#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main(int argc, char** argv) {
  // define total data elements
  int nElem=1024;

  // define grid and block structure
  dim3 block (1024);
  dim3 grid ((nElem+block.x-1)/block.x);
  cout << "grid.x " << grid.x << " block.x " << block.x << endl;

  // reset block
  block.x=512;
  grid.x=(nElem+block.x-1)/block.x;
  cout << "grid.x " << grid.x << " block.x " << block.x << endl;

  // reset block
  block.x=256;
  grid.x=(nElem+block.x-1)/block.x;
  cout << "grid.x " << grid.x << " block.x " << block.x << endl;

  // reset block
  block.x=128;
  grid.x=(nElem+block.x-1)/block.x;
  cout << "grid.x " << grid.x << " block.x " << block.x << endl;

  // reset device before you leave
  cudaDeviceReset();
  return 0;
}
