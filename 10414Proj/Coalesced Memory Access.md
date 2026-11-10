Global memory is accessed via 32-, 64-, or 128-byte naturally aligned transactions

When the 32 threads in a [[warp]] access contiguous, aligned memory addresses, the hardware coalesces them into a single memory transaction 

In fused kernels, input tensors are loaded in coalesced tiles into shared memory before performing mathematical operations. 

