### Python Interface
ops_mathematic.py - define the forward and gradient operations for each kind of operation 
* These are all written in python so I imagine this will spawn different CUDA processes for handling 

[[autograd.py]] - topological computational graph support 
* Take gradient of output node with respect to each node in node_list


[[ndarray.py]]


[[ndarray_backend_cuda.cu]]

needle.Tensor 

## Goal
trace a single existing Needle operation all the way from Python through its CUDA backend. Then implement a trivial fused operation, such as `a * sigmoid(a) * b`.

Transformer - fused kernel project 
fused RMSNorm, RoPE, SwiGLU, and cross-entropy, with a small dense Transformer as the evaluation model.


MoE: MoE layer in Needle and optimize token dispatch, grouping, or expert computation.


## Implementation 
Write GPU device functions (like __global___ ) in CUDA C++ for both the forward and backward gradient computation 

Design the kernel logic to stage into SRAM or registers 
Instead of materializing large intermediate tensors to main VRAM, store only minimal summary statistics (such as $RMS(x)$, row maxima, or LogSumExp) needed to recompute intermediate steps during the backward pass

**Launch Execution Configuration:** Compute dynamic grid dimensions, thread block layouts, and shared memory allocations (`<<

Encapsulate the custom autograd functions 








