# Chapter3
## Introduction
### GPU Architecture
Streaming Multiprocessors (SM)

Fermi SM:

- CUDA Cores
- SharedMemory/L1 Cache
- Register File
- Load/Store Units
- Special Function Units
- Warp Schedule

- Single Instruction Multiple Thread (SIMT) architechture to mange and execute threads in groups of 32 called warps
- Single Instruction Multiple Data (SIMD)

A thread block is scheduled on only one SM.
- Shared Memory - thread blocks on the SM
- registers - threads

### The Fermi Architecture
512 total CUDA cores

- Each core has an Arithmetic logic unige (ALU) and a floating-point unit (FPU)
- 16 SM, each with 32 CUDA cores
- six 384-bit GDDR5 DRAM memory
- Connects GPU via the PCI Express bus
- Coherent 768 KB L2 cache

### The Kepler Architecture
- Enhanced SMs
- Dynamic Parallelism
  - Kepler GPUs allow the GPU to dynamically launch new grids
- Hyper-Q
 - simultaneous hardware connections btw the CPU and GPU, enabling CPU cores to simultaneously run more tasks on the GPU

### Pascal and Maxwell
[URL (in Japanese)](https://pc.watch.impress.co.jp/docs/column/kaigai/755994.html)

### Profile-Driven Optimization
- `nvprof`
- `nvvp`: Visual Profiler

## Understandint the Nature of Warp Execution
### Warps and Thread Blocks
32 threads for a warp

### Warp Divergence
Warp divergence(: threads in the same warp executing different instructions) would cause a paradox

- Warp divergence occurs when threads within a warp take different code paths
- Different `if-then-else` branches are executed serially
- Try to adjust branch granularity to be a multiple of warp size to avoid warp divergence
- Different warps can execute different code with no penalty on performance

### Resource Partitioning
Resources

- Program counters
- Registers (per SM) <-> threads(warps)
- Shared memory (per SM) <-> thread blocks
- active block
- active warps
  - Selected warp: actively executing
  - Eligible warp: ready for execution but not currently executing
  - Stalled warp: not ready for execution

### Latency Hiding
~~~
Number of Required Warps = Latency * Throughput
~~~
a thoeretically peak value, Thoughput: an achieved value

Increase parallesim

- __Instruction-level parallelism (ILP)__: More independent instructions within a thread
- __Thread-level parallelism (TLP)__: More concurrently eligible threads

### Occupancy
```
occupancy = active warps / maximum warps
```
Guidelines for grid and block size

- Keep the number of threads per block a multiple of warp size (32)
- Avoid small block sizes: Start with at least 128 or 256 threads per block
- Adjust block size up or down according to kernel resource requirements
- Keep the number of blocks much greater than the number of SMs to expose sufficient parallelism to your device
- Conduct experiments to dixcover the best execution configuration and resource usage

### Synchronization

- __System-level__: Wait for all work on both the host and the device to complete
- __Block-level__: Wait for all threads in a thread block to reach the same point in execution on the device

### Scalability
transparent scalability: the ability to execute the same application code on a varying number of compute cores

## Exposing Parallelism

### Checking Active Warps with nvprof

`nvprof --metrics achieved_occupancy sumMatrix 32 32`: the ratio of the average active warps per cycle to the maximum number of warps supported on an SM

Trade-off: number of active warps vs. occupancy

### Checking Memory Operations with nvprof

`nvprof --metrics gld_throughput./sumMatrix 32 32`: to check the memory read efficiency

`gld_efficiency`: the ratio of requested global load throughput to required global load throughput

### Exposing More Parallelism


