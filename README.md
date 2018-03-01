# Learning CUDA

## Hello, World! (dir:hello)
~~~
nvcc hello_world.cu
./a.out
~~~

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
[CUDA toolkit documentation (Thrust)](http://docs.nvidia.com/cuda/thrust/index.html)

* thrust1: (Vector) host_vector, device_vector
* thrust2: (Vector) copy, fill, sequence
* thrust3: (Algorithms;Transformations) replace, transform, functional
* thrust4: (Algorithms;Reductions) transform_reduce;to reduce an input sequence to a single value

# Professional CUDA C Programming
[url of pdf](http://www.hds.bme.hu/~fhegedus/C++/Professional%20CUDA%20C%20Programming.pdf)
