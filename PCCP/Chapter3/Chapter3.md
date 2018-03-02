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
```math
Number of Required Warps = Latency * Throughput
```
a thoeretically peak value, Thoughput: an achieved value

Increase parallesim
- Instruction-level parallelism (ILP): More independent instructions within a thread
- Thread-level parallelism (TLP): More concurrently eligible threads

### Occupancy
```math
occupancy = \frac{active warps}{maximum warps}
```
