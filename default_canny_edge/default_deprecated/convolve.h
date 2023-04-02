#include <iostream>

void convolve(double*** matrix, double*** image, double*** kernel, int imgHeight, int imgWidth, int kernelHeight, int kernelWidth);

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

