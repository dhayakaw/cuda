[url](https://github.com/atinfinity/lab/wiki/OpenCVビルド情報、CUDA情報取得)

# use OpenCV with CUDA
[url](https://alisentas.com/image%20processing/build-opencv-3-4-0-with-cuda-and-tbb-support-in-arch-linux)
Download opencv-3.4.1 and opencv_contrib-3.4.1
```
cd ~/Downloads/opencv-3.4.1/
mkdir build
cd build
export CXXFLAGS="-std=c++11"
export CXX=/opt/cuda/bin/g++
export CC=/opt/cuda/bin/gcc
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=~/opencv -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.1/modules -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES -std=c++11 --expt-relaxed-constexpr" -D BUILD_opencv_java=OFF -DCUDA_GENERATION=Maxwell -DBUILD_opencv_python=OFF -DBUILD_opencv_python2=OFF -DWITH_OPENMP=ON -DBUILD_DOCS=OFF ..
```
