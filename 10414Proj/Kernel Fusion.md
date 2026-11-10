
30m - read + understand 
60m - attempt + document

Eliminates memory bandwidth bottlenecks during transformer training 
* Many transformer operations are memory bound, meaning that they are bottlenecked by this cycle instead of by the operation itself (data read and write to high bandwidth memory)

[[High Bandwidth Memory]] - Main GPU memory is large but relatively slow

[[On Chip SRAMs]]



## Problems with Eager pytorch execution 

Each step or operation in Python launches an independent C++ / [[CUDA]] kernel on the GPU 
* Each kernel executes a fixed [[cycle]]
[[Intermediate Tensor]] operations are repeatedly written to HBM and re-read by the next kernel 
* These intermediate activations, during training, must also be kept in HBM for the [[backward pass]]
* this causes peak [[High Bandwidth Memory]] usage


## Kernel Fusion 
Combines multiple sequential operations into a single GPU kernel 
* GPU loads the initial input tensor onto [[On Chip SRAMs]] once and performs all the intermediate mathematical operations on-chip ([[registers]]
* ) and writes the output tensor back into the [[High Bandwidth Memory]] once 

See: [[Softmax Operation]]

