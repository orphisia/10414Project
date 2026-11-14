#include <cuda_runtime.h>
#include <math.h>


__global__ void fwd_swiGlu(
    const float* x1, 
    const float* x2, 
    float* Y,
    int total_elements
){ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    // prevent the threads from oob accesses 
    if (idx < total_elements){ 
        // load input values 
        float a = x1[idx];
        float b = x2[idx];

        // calculate sigmoid 
        float sigmoid_a = 1.0f/(1.0f + expf(-a));
        

        //calculate siLU(a)
        float silu_a = a * sigmoid_a; 

        Y[idx] = silu_a * b; 
    }
    }

__global__ void bwd_swiGlu(
    const float* x1, 
    const float* x2, 
    const float* dY, 
    float* dX1,
    float* dX2, 
    int total_elements
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < total_elements){
        // load the upstream gradient and inputs 
        float dy = dY[idx];
        float a = x1[idx];
        float b = x2[idx];

        // recompute values from the forward
        // note recomputation for the benchmarks  
        float sigmoid_a = 1.0f/(1.0f + expf(-a));
        float silu_a = a * sigmoid_a; 

        // gradient wrt x_2 
        dX2[idx] = silu_a * dy; 

        //derivative of silu 
        float dsilu_da = sigmoid_a * (1.0f + a * (1.0f - sigmoid_a));

        // gradient wrt x_1 
        dX1[idx] = dy * b * dsilu_a;

    }
}

void launch_fused_swiGlu_forward(
    const float* x1, 
    const float* x2, 
    float* Y, 
    int total_elements
){
   if (total_elements == 0){
        return; 
    }    
   int threads = 256; 
   int blocks = (total_elements + threads - 1) / threads; 

    fwd_swiGlu<<<blocks, threads>>>(
        x1, x2, Y, total_elements
    );
}

void launch_fused_swiGlu_backward(
    const float* x1,
    const float* x2,
    const float* dY, 
    float* dX1,
    float* dX2, 
    int total_elements
){
    if (total_elements == 0){
        return;
    }
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    bwd_swiGlu<<<(blocks, threads)>>>(
        dY, x1, x2, dX1, dX2, total_elements
    ); 
}
