# Learning CUDA

## Matrix calculation with CPU or GPU (dir:matrix)
~~~
nvcc -o matrix_gpu matrix_gpu.cu -I /opt/cuda/samples/common/inc
~~~

~~~
time ./matrix_cpu 
Processing time: 7 (sec)
./matrix_cpu  6.37s user 0.01s system 99% cpu 6.380 total
time ./matrix_gpu 
Processing time: 232.233826 (msec)
./matrix_gpu  0.17s user 0.20s system 99% cpu 0.373 total
~~~

GPU is 37 times faster than CPU!

## Thrust (dir:thrust)
[CUDA toolkit documentation](http://docs.nvidia.com/cuda/thrust/index.html)

* thrust1: host_vector, device_vector
* thrust2: copy, fill, sequence
* thrust3: replace transform functional
