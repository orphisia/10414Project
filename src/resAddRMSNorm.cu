#include <cuda_runtime.h>

__global__ void fused_res_add_rms_norm( 
                                const float *X,
                                const float *R,
                                const float *gamma,
                                float *Z,
                                float *Y,
                                int num_tokens,
                                int hidden_dim,
                                float epsilon) {
    
    // Array shared within each block to store partial sums,
    // which will later be reduced
    extern __shared__ float partial_sums[];
    __shared__ float token_sum;
    __shared__ float inv_rms;

    int row_start = blockIdx.x * hidden_dim; // Each block handles one token, this index is per block
    float sum_Z_squared = 0.0f; // Element-wise sum of element-wise Z**2
    
    // Computing the residual Z, and its squared sum
    for (int i=threadIdx.x; i<hidden_dim; i+= blockDim.x) { // (hidden_dim / thread count) elements per thread (coalesced concurrent accesses)
        int idx = row_start + i;
        float z_idx = X[idx] + R[idx]; // compute z_idx
        Z[idx] = z_idx;
        sum_Z_squared += z_idx * z_idx; // add z_idx**2
    }

    partial_sums[threadIdx.x] = sum_Z_squared; 
    // Before performing reduction, wait until all threads have written to partial_sums
    __syncthreads();

    // Performing reduction and addition on the squared sum Z
    // Folding halves, final result stored in index 0
    for (int j= blockDim.x / 2; j>0; j /= 2) {
        if (threadIdx.x < j) {
            partial_sums[threadIdx.x] += partial_sums[j + threadIdx.x];
        }
        __syncthreads();
    }

    // Let one thread update both shared variables
    if (threadIdx.x == 0) {
        token_sum = partial_sums[0];
        inv_rms = rsqrtf((token_sum / hidden_dim) + epsilon);
    }
    __syncthreads();

    // Update Y to contain the RMS-normalized residual
    for (int k=threadIdx.x; k<hidden_dim; k+= blockDim.x) {
        int kdx = row_start + k;
        float z_kdx = Z[kdx]; // compute z_kdx
        float y_kdx = gamma[k] * z_kdx  * inv_rms;
        Y[kdx] = y_kdx;
    }
}

void launch_fused_res_add_rms_norm(
    const float* X,
    const float* R,
    const float* gamma,
    float* Z,
    float* Y,
    int num_tokens,
    int hidden_dim,
    float epsilon
) {
    int threads = 256;
    size_t shared_bytes = threads * sizeof(float);

    fused_res_add_rms_norm<<<num_tokens, threads, shared_bytes>>>(
        X, R, gamma, Z, Y,
        num_tokens, hidden_dim, epsilon
    );
}

