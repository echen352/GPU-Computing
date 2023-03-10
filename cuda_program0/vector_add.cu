#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

__global__ void vector_add(int *out, int *a, int *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    	out[tid] = a[tid] + b[tid];
}

int main(int argc, char **argv){
    int *h_a, *h_b, *h_out;
    int *d_a, *d_b, *d_out;
    int N = atoi(argv[1]);
    clock_t t;

    // Allocate host memory
    h_a   = (int*)malloc(sizeof(int) * N);
    h_b   = (int*)malloc(sizeof(int) * N);
    h_out = (int*)malloc(sizeof(int) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        h_a[i] = i;
        h_b[i] = i + 1;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(int) * N);
    cudaMalloc((void**)&d_b, sizeof(int) * N);
    cudaMalloc((void**)&d_out, sizeof(int) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * N, cudaMemcpyHostToDevice);
    
    t = clock();
    
    // Executing kernel
    int blockSize = 256;
    int gridSize = ((N + blockSize) / blockSize);
    vector_add<<<gridSize,blockSize>>>(d_out, d_a, d_b, N);
    
    t = clock() - t;
    
    // Transfer data back to host memory
    cudaMemcpy(h_out, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);
    
    // Print results
    /*for (int i = 0; i < N; i++) {
    	printf("\n C[%d]=%d", i, h_out[i]);
    }*/
    
    printf("\nTime Taken: %f sec\n", ((double)t)/CLOCKS_PER_SEC);
    
    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(h_a); 
    free(h_b); 
    free(h_out);
}
