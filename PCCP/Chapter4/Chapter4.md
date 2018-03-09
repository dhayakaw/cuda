# Chapter4: Global Memory

## Introduction

### Memory Hierarchy

Typical hierarchy
Registers -> Caches -> Main memory -> Disk memory
Speed: fast -> slow
Size: small -> big

### CUDA Memory Model

Figure 4-2 (p.138)

* Registers
* Local memory
  * only used when variables cannot fit into the register space. High latency, low bandwidth
* Shared memory
* Constant memory
* Texture memory
* Global memory

### Memory Transfer

* GDDR5: 144 GB/s
* PCIe: 8GB/s

### Pinned Memory

`cudaMallocHost`

### Zero-Copy Memory


