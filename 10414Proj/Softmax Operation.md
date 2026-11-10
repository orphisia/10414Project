$$y_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

## Unfused execution (standard safe)

Kernel 1 - find max of x by reading vector x from HBM , computes max, then writes it back to HBM 

Kernel 2 - exponentials Reads x and max from the HBM,, write intermediate exponentials back to HBM 

This continues for multiple kernels 

_Total HBM Traffic:_ **4 passes (reads/writes)** per vector element across 4 separate kernel launche


Fused Execution (Online Softmax Kernel):
_Single Kernel:_ Loads vector $x$ from HBM into fast SRAM **once** $\rightarrow$ dynamically updates the running maximum and exponential sum on-chip $\rightarrow$ divides on-chip $\rightarrow$ writes final output vector $y$ to HBM **once**


1 read and 1 write per statement 

This in general will help to increase throughput by keeping the operations on [[On Chip SRAMs]] and not having to wait for expensive read/ write operations 

(idling time for memory transfers)