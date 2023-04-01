__global__ void cuda_convolve(double* matrix, double* image, double* mask, int imgHeight, int imgWidth, int maskHeight, int maskWidth);

__global__ void cuda_convolve(double* matrix, double* image, double* mask, int imgHeight, int imgWidth, int maskHeight, int maskWidth) {
    double sum;
    int k, m, offseti, offsetj;
    int local_i, local_j, global_i, global_j;
    
    extern __shared__ double Ashared[];
    
    // map local and global IDs
    local_i = threadIdx.x;
    local_j = threadIdx.y;
    global_i = blockIdx.x * blockDim.x + threadIdx.x;
    global_j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // set shared memory
    Ashared[local_i * blockDim.y + local_j] = image[global_i * imgWidth + global_j];
    __syncthreads();
    
    if (global_i < imgHeight && global_j < imgWidth) {
	    sum = 0;
	    for (k = 0; k < maskHeight; k++) {
		for (m = 0; m < maskWidth; m++) {
		    offseti = -1 * (maskHeight / 2) + k;
		    offsetj = -1 * (maskWidth / 2) + m;
		    if ((local_i + offseti) > -1 && (local_i + offseti) < blockDim.x && (local_j + offsetj) > -1 && (local_j + offsetj) < blockDim.y)
		    	// entries can be accessed from shared memory
		            sum += Ashared[(local_i + offseti) * blockDim.y + local_j + offsetj] * mask[k * maskWidth + m];
		    else if ((global_i + offseti) > -1 && (global_i + offseti) < imgHeight && (global_j + offsetj) > -1 && (global_j + offsetj) < imgWidth)
		    	// out of bounds of shared memory, retrieve entries from L2 cache
		    	sum += image[(global_i + offseti) * imgWidth + global_j + offsetj] * mask[k * maskWidth + m];
		}
	    }
	    matrix[global_i * imgWidth + global_j] = sum;
    }

    return;
}

