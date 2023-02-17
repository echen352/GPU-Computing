#include <iostream>

double** convolve(double** image, double** kernel, int imgHeight, int imgWidth, int kernelHeight, int kernelWidth);
double** allocateMatrix(int rows, int cols);

double** convolve(double** image, double** kernel, int imgHeight, int imgWidth, int kernelHeight, int kernelWidth) {
	double** outputMatrix;
    double sum;
    int offseti, offsetj;

    outputMatrix = allocateMatrix(imgHeight, imgWidth);

    for (int i = 0; i < imgHeight; i++) {
        for (int j = 0; j < imgWidth; j++) {
            sum = 0;
            for (int k = 0; k < kernelHeight; k++) {
                for (int m = 0; m < kernelWidth; m++) {
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
    newMatrix = (double**)malloc(sizeof(double*) * rows);
    if (newMatrix == NULL) {
        std::cout << "Error allocating memory" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        newMatrix[i] = (double*)malloc(sizeof(double) * cols);
        if (newMatrix[i] == NULL) {
            std::cout << "Error allocating memory" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            newMatrix[i][j] = 0;
        }
    }

    return newMatrix;
}