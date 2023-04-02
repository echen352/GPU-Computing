#include <iostream>
#include <cmath>
#include "convolve.h"

class Gradient
{
public:
	Gradient();
	double** horizontal;
	double** vertical;
	double** magnitude;
	double** gradient;
	void horizontalGradient(double** image, double* gauss, double* gaussDeriv, int imgHeight, int imgWidth, int gaussLength);
	void verticalGradient(double** image, double* gauss, double* gaussDeriv, int imgHeight, int imgWidth, int gaussLength);
	void magnitudeGradient(double** vertical, double** horizontal, int height, int width);
	double** transposeToVertical(double* matrix, int height, int width);
	double** transposeToHorizontal(double* matrix, int height, int width);
	double** reverseSign(double** matrix, int height, int width, int dir);
	double** allocateGradientMatrix(int height, int width);
	void deallocateMatrix(int rows);
};

Gradient::Gradient() {
	horizontal = NULL;
	vertical = NULL;
	magnitude = NULL;
	gradient = NULL;
}

void Gradient::horizontalGradient(double** image, double* gauss, double* gaussDeriv, int imgHeight, int imgWidth, int gaussLength) {
	double** verticalGauss;
	double** tempHorizontal;
	double** horizontalGaussDeriv;
	double** flippedGaussDeriv;
	int i;
	
	verticalGauss = transposeToVertical(gauss, gaussLength, 1);

	horizontalGaussDeriv = allocateGradientMatrix(1, gaussLength);

	#pragma omp parallel for num_threads(NUMTHREADS)
	for (i = 0; i < gaussLength; i++)
		horizontalGaussDeriv[0][i] = gaussDeriv[i];

	tempHorizontal = convolve(image, verticalGauss, imgHeight, imgWidth, gaussLength, 1);

	flippedGaussDeriv = reverseSign(horizontalGaussDeriv, 1, gaussLength, 0);

	this->horizontal = convolve(tempHorizontal, flippedGaussDeriv, imgHeight, imgWidth, 1, gaussLength);
	
	for (i = 0; i < gaussLength; i++)
		free(verticalGauss[i]);
	free(verticalGauss);

	for (i = 0; i < imgHeight; i++)
		free(tempHorizontal[i]);
	free(tempHorizontal);

	for (i = 0; i < 1; i++)
		free(horizontalGaussDeriv[i]);
	free(horizontalGaussDeriv);

	for (i = 0; i < 1; i++)
		free(flippedGaussDeriv[i]);
	free(flippedGaussDeriv);
	
	return;
}

void Gradient::verticalGradient(double** image, double* gauss, double* gaussDeriv, int imgHeight, int imgWidth, int gaussLength) {
	double** horizontalGauss;
	double** tempVertical;
	double** verticalGaussDeriv;
	double** flippedGaussDeriv;
	int i;

	horizontalGauss = transposeToHorizontal(gauss, 1, gaussLength);

	verticalGaussDeriv = transposeToVertical(gaussDeriv, gaussLength, 1);

	tempVertical = convolve(image, horizontalGauss, imgHeight, imgWidth, 1, gaussLength);

	flippedGaussDeriv = reverseSign(verticalGaussDeriv, gaussLength, 1, 1);
				
	this->vertical = convolve(tempVertical, flippedGaussDeriv, imgHeight, imgWidth, gaussLength, 1);

	for (i = 0; i < 1; i++ )
		free(horizontalGauss[i]);
	free(horizontalGauss);

	for (i = 0; i < imgHeight; i++)
		free(tempVertical[i]);
	free(tempVertical);

	for (i = 0; i < gaussLength; i++)
		free(verticalGaussDeriv[i]);
	free(verticalGaussDeriv);

	for (i = 0; i < gaussLength; i++)
		free(flippedGaussDeriv[i]);
	free(flippedGaussDeriv);
	
	return;
}

void Gradient::magnitudeGradient(double** vertical, double** horizontal, int height, int width) {
	double verticalSquare;
	double horizontalSquare;
	int x, y;

	this->magnitude = allocateGradientMatrix(height, width);
	this->gradient = allocateGradientMatrix(height, width);
	
	# pragma omp parallel for private(y, verticalSquare, horizontalSquare) num_threads(NUMTHREADS)
	for (x = 0; x < height; x++) {
		for (y = 0; y < width; y++) {
			verticalSquare = (vertical[x][y])*(vertical[x][y]);
			horizontalSquare = (horizontal[x][y]) * (horizontal[x][y]);
			this->magnitude[x][y] = sqrt(verticalSquare + horizontalSquare);
			this->gradient[x][y] = atan2(horizontal[x][y], vertical[x][y]);
		}
	}

	return;
}

double** Gradient::transposeToVertical(double* matrix, int height, int width) {
	double** transposedMatrix;
	int i;

	transposedMatrix = allocateGradientMatrix(height, width);

	for (i = 0; i < height; i++)
		transposedMatrix[i][0] = matrix[i];
		
	return transposedMatrix;
}

double** Gradient::transposeToHorizontal(double* matrix, int height, int width) {
	double** transposedMatrix;
	int i;

	transposedMatrix = allocateGradientMatrix(height, width);

	for (i = 0; i < width; i++)
		transposedMatrix[0][i] = matrix[i];
		
	return transposedMatrix;
}

double** Gradient::reverseSign(double** matrix, int height, int width, int dir) {
	double** reversedMatrix;
	int i;
	
	reversedMatrix = allocateGradientMatrix(height, width);

	if (dir == 1) {
		for (i = 0; i < height; i++)
			reversedMatrix[i][0] = matrix[i][0] * -1;
	}
	else if (dir == 0) {
		for (i = 0; i < width; i++)
			reversedMatrix[0][i] = matrix[0][i] * -1;
	}

	return reversedMatrix;
}

double** Gradient::allocateGradientMatrix(int height, int width) {
	double** newMatrix;
	int i;

	newMatrix = (double**)malloc(sizeof(double*)* height);
	if (newMatrix == NULL) {
		std::cout << "Error allocating memory" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < height; i++) {
		newMatrix[i] = (double*)malloc(sizeof(double) * width);
		if (newMatrix[i] == NULL) {
			std::cout << "Error allocating memory" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	return newMatrix;
}

void Gradient::deallocateMatrix(int rows) {
	int i;
	
	for (i = 0; i < rows; i++) {
		free(this->gradient[i]);
		free(this->magnitude[i]);
		free(this->horizontal[i]);
		free(this->vertical[i]);
	}
	free(this->gradient);
	free(this->magnitude);
	free(this->horizontal);
	free(this->vertical);

	return;
}
