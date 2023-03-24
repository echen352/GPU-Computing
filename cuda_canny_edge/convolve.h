#include <iostream>

//__global__ void convolve(double*** matrix, double*** image, double*** kernel, int imgHeight, int imgWidth, int kernelHeight, int kernelWidth);

//void convolve(double*** matrix, double*** image, double*** kernel, int imgHeight, int imgWidth, int kernelHeight, int kernelWidth);

void convolve(double*** matrix, double*** image, double*** kernel, int imgHeight, int imgWidth, int kernelHeight, int kernelWidth) {
    double sum;
    int offseti, offsetj;

    for (int i = 0; i < imgHeight; i++) {
        for (int j = 0; j < imgWidth; j++) {
            sum = 0;
            for (int k = 0; k < kernelHeight; k++) {
                for (int m = 0; m < kernelWidth; m++) {
                    offseti = -1 * floor(kernelHeight / 2) + k;
                    offsetj = -1 * floor(kernelWidth / 2) + m;
                    if ((i + offseti) > -1 && (i + offseti) < imgHeight) {
                        if ((j + offsetj) > -1 && (j + offsetj) < imgWidth) {
                            sum += (*image)[i + offseti][j + offsetj] * (*kernel)[k][m];
                        }
                    }
                }
            }
           (*matrix)[i][j] = sum;
        }
    }

    return;
}

__global__ void cuda_convolve(double** matrix, double** image, double** mask, int imgHeight, int imgWidth, int maskHeight, int maskWidth) {
    double sum = 0;
    int k, m, offseti, offsetj;
    int local_i, local_j, global_i, global_j;
    
    extern __shared__ double A[];
    
    local_i = threadIdx.x;
    local_j = threadIdx.y;
    global_i = blockIdx.x * blockDim.x + local_i;
    global_j = blockIdx.y * blockDim.y + local_j;
    
    A[local_i * blockDim.y + local_j] = image[global_i][global_j];
    __syncthreads();
    
    if (global_i < imgHeight && global_j < imgWidth) {
	    sum = 0;
	    for (k = 0; k < maskHeight; k++) {
		for (m = 0; m < maskWidth; m++) {
		    offseti = -1 * maskHeight / 2 + k;
		    offsetj = -1 * maskWidth / 2 + m;
		    if ((local_i + offseti) > -1 && (local_i + offseti) < blockDim.x && (local_j + offsetj) > -1 && (local_j + offsetj) < blockDim.y)
		    	// entries can be accessed from shared memory
		            sum += A[(local_i + offseti) * blockDim.y + (local_j + offsetj)] * mask[k][m];
		    else if ((global_i + offseti) > -1 && (global_i + offseti) < imgHeight && (global_j + offsetj) > -1 && (global_j + offsetj) < imgWidth)
		    	// out of bounds of shared memory, retrieve entries from L2 cache
		    	sum += image[global_i + offseti][global_j + offsetj] * mask[k][m];
		}
	    }
	    matrix[global_i][global_j] = sum;
    }

    return;
}

