Learning CUDA
`nvcc -o matrix_gpu matrix_gpu.cu -I /opt/cuda/samples/common/inc`

~~~
time ./matrix_cpu 
Processing time: 7 (sec)
./matrix_cpu  6.37s user 0.01s system 99% cpu 6.380 total
time ./matrix_gpu 
Processing time: 232.233826 (msec)
./matrix_gpu  0.17s user 0.20s system 99% cpu 0.373 total
~~~

37 times faster!
