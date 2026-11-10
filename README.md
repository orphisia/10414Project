# Fused Heterogenous Transformer Support

## Project Description:
We are extending needle with support for fused CUDA and Apple Metal transformer kernels along with support for selectively composing them, by specifying the desired target libraries through the needle interface. Namely, we will contribute with implementations of:
- ResidualAdd + RMSNorm;
- QK RMSNorm + RoPE;
- SwiGLU;
- Cross-entropy;
  
As an extension, after implementing these kernels we will optionally add a functionality that processes some compute graph and replaces groups of computation nodes with their corresponding fused kernels where there is a match. The implementation of this would borrow from compiler techniques such as constant folding and propagation.

## Challenges: 
Optimizing these kernels on both CUDA and Apple metal will require an understanding of tradeoffs in parallelism (such as communication vs work splitting), as well as contrastive reasoning regarding the features of CUDA compared to Apple Metal, as we will probably begin from a CUDA implementation then work up to an Apple Metal version. 
Additionally, if we choose to pursue the extension we will process the compute graph in a way that is verifiably correct. 

## Action Plan: TBD



