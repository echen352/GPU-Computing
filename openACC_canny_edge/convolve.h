#include <stdio.h>

void convolve_1d(double* matrix, double* image, double* kernel, int imgHeight, int imgWidth, int kernelHeight, int kernelWidth);

// convolutions work only using 1D masks
void convolve_1d(double* matrix, double* image, double* kernel, int imgHeight, int imgWidth, int kernelHeight, int kernelWidth) {
    double sum;
    int offseti, offsetj, i, j, k, m;
    
    #pragma acc parallel copyin(image[0:imgHeight*imgWidth]) copyin(kernel[0:kernelHeight*kernelWidth]) copyout(matrix[0:imgHeight*imgWidth]) private(i, j, k, m, offseti, offsetj) num_gangs(1024) num_workers(32) vector_length(1024)
    {
    #pragma acc loop gang worker
    for (i = 0; i < imgHeight; i++) {
	for (j = 0; j < imgWidth; j++) {
	    sum = 0;
	    #pragma acc loop vector
	    for (k = 0; k < kernelHeight; k++) {
	        for (m = 0; m < kernelWidth; m++) {
	            offseti = -1 * floor(kernelHeight / 2) + k;
	            offsetj = -1 * floor(kernelWidth / 2) + m;
	            if ((i + offseti) > -1 && (i + offseti) < imgHeight) {
	                if ((j + offsetj) > -1 && (j + offsetj) < imgWidth) {
	                    sum += image[(i + offseti) * imgWidth + j + offsetj] * kernel[k * kernelWidth + m];
	                }
	            }
	        }
	    }
	   matrix[i * imgWidth + j] = sum;
	}
    }
    }

    return;
}

