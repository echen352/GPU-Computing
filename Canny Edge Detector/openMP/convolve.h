#include <iostream>
#include <omp.h>

double** convolve(double** image, double** kernel, int imgHeight, int imgWidth, int kernelHeight, int kernelWidth);
double** allocateMatrix(int rows, int cols);

double** convolve(double** image, double** kernel, int imgHeight, int imgWidth, int kernelHeight, int kernelWidth) {
	double** outputMatrix;
	double sum;
	int offseti, offsetj, i, j, k, m;

	outputMatrix = allocateMatrix(imgHeight, imgWidth);
	
	#pragma omp parallel for private(offseti, offsetj, sum, j, k, m) num_threads(NUMTHREADS)
	for (i = 0; i < imgHeight; i++) {
	    for (j = 0; j < imgWidth; j++) {
		sum = 0;
		for (k = 0; k < kernelHeight; k++) {
		    for (m = 0; m < kernelWidth; m++) {
			offseti = -1 * floor(kernelHeight / 2) + k;
			offsetj = -1 * floor(kernelWidth / 2) + m;
			if ((i + offseti) > -1 && (i + offseti) < imgHeight) {
			    if ((j + offsetj) > -1 && (j + offsetj) < imgWidth) {
			        sum += image[i + offseti][j + offsetj] * kernel[k][m];
			    }
			}
		    }
		}
		outputMatrix[i][j] = sum;
	    }
	}

	return outputMatrix;
}

double** allocateMatrix(int rows, int cols) {
	double** newMatrix;
	int i, j;
    
	newMatrix = (double**)malloc(sizeof(double*) * rows);
	if (newMatrix == NULL) {
	std::cout << "Error allocating memory" << std::endl;
	exit(EXIT_FAILURE);
	}

	for (i = 0; i < rows; i++) {
	    newMatrix[i] = (double*)malloc(sizeof(double) * cols);
	    if (newMatrix[i] == NULL) {
		std::cout << "Error allocating memory" << std::endl;
		exit(EXIT_FAILURE);
	    }
	}
	
	# pragma omp parallel for private(j) num_threads(NUMTHREADS)// collapse(2)
	for (i = 0; i < rows; i++) {
	    for (j = 0; j < cols; j++) {
		newMatrix[i][j] = 0;
	    }
	}

	return newMatrix;
}
