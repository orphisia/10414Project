1D, 2D or 3D collection of up to 1024 CUDA [[Thread]] co-scheduled to run on a single [[Streaming Multiprocessor]]

[[Thread]] within the same block can share data through on-chip memory and sync execution 

Thread blocks across a grid execute independently in any order 

Enables cuda programs to scale independently regardless across GPUs of the number of SMs 

Loads a chunk of tensor data from VRAM using coalesced memory accesses, and this data is then kept in shared memory ([[On Chip SRAMs]]) or registers 
* Threads apply elementwise math directly in registers
* the finalized tensor is written in one coalesced write. 

