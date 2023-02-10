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

	verticalGauss = transposeToVertical(gauss, gaussLength, 1);

	tempHorizontal = convolve(image, verticalGauss, imgHeight, imgWidth, gaussLength, 1);

	horizontalGaussDeriv = allocateGradientMatrix(1, gaussLength);
	for (int i = 0; i < gaussLength; i++)
		horizontalGaussDeriv[0][i] = gaussDeriv[i];

	flippedGaussDeriv = reverseSign(horizontalGaussDeriv, 1, gaussLength, 0);

	this->horizontal = convolve(tempHorizontal, flippedGaussDeriv, imgHeight, imgWidth, 1, gaussLength);

	for (int i = 0; i < gaussLength; i++)
		free(verticalGauss[i]);
	free(verticalGauss);

	for (int i = 0; i < imgHeight; i++)
		free(tempHorizontal[i]);
	free(tempHorizontal);

	for (int i = 0; i < 1; i++)
		free(horizontalGaussDeriv[i]);
	free(horizontalGaussDeriv);

	for (int i = 0; i < 1; i++)
		free(flippedGaussDeriv[i]);
	free(flippedGaussDeriv);
	return;
}

void Gradient::verticalGradient(double** image, double* gauss, double* gaussDeriv, int imgHeight, int imgWidth, int gaussLength) {
	double** horizontalGauss;
	double** tempVertical;
	double** verticalGaussDeriv;
	double** flippedGaussDeriv;

	horizontalGauss = transposeToHorizontal(gauss, 1, gaussLength);

	tempVertical = convolve(image, horizontalGauss, imgHeight, imgWidth, 1, gaussLength);

	verticalGaussDeriv = transposeToVertical(gaussDeriv, gaussLength, 1);

	flippedGaussDeriv = reverseSign(verticalGaussDeriv, gaussLength, 1, 1);

	this->vertical = convolve(tempVertical, flippedGaussDeriv, imgHeight, imgWidth, gaussLength, 1);

	for (int i = 0; i < 1; i++ )
		free(horizontalGauss[i]);
	free(horizontalGauss);

	for (int i = 0; i < imgHeight; i++)
		free(tempVertical[i]);
	free(tempVertical);

	for (int i = 0; i < gaussLength; i++)
		free(verticalGaussDeriv[i]);
	free(verticalGaussDeriv);

	for (int i = 0; i < gaussLength; i++)
		free(flippedGaussDeriv[i]);
	free(flippedGaussDeriv);
	return;
}

void Gradient::magnitudeGradient(double** vertical, double** horizontal, int height, int width) {
	double verticalSquare;
	double horizontalSquare;

	this->magnitude = allocateGradientMatrix(height, width);
	this->gradient = allocateGradientMatrix(height, width);

	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
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

	transposedMatrix = allocateGradientMatrix(height, width);

	for (int i = 0; i < height; i++) {
		transposedMatrix[i][0] = matrix[i];
		std::cout << transposedMatrix[i][0] << std::endl;
	}
	return transposedMatrix;
}

double** Gradient::transposeToHorizontal(double* matrix, int height, int width) {
	double** transposedMatrix;

	transposedMatrix = allocateGradientMatrix(height, width);

	for (int i = 0; i < width; i++) {
		transposedMatrix[0][i] = matrix[i];
		std::cout << transposedMatrix[0][i] << std::endl;
	}
	return transposedMatrix;
}

double** Gradient::reverseSign(double** matrix, int height, int width, int dir) {
	double** reversedMatrix;
	reversedMatrix = allocateGradientMatrix(height, width);

	if (dir == 1) {
		for (int i = 0; i < height; i++) {
			reversedMatrix[i][0] = matrix[i][0] * -1;
		}
	}
	else if (dir == 0) {
		for (int i = 0; i < width; i++) {
			reversedMatrix[0][i] = matrix[0][i] * -1;
		}
	}

	return reversedMatrix;
}

double** Gradient::allocateGradientMatrix(int height, int width) {
	double** newMatrix;

	newMatrix = (double**)malloc(sizeof(double*)* height);
	if (newMatrix == NULL) {
		std::cout << "Error allocating memory" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < height; i++) {
		newMatrix[i] = (double*)malloc(sizeof(double) * width);
		if (newMatrix[i] == NULL) {
			std::cout << "Error allocating memory" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	return newMatrix;
}

void Gradient::deallocateMatrix(int rows) {
	for (int i = 0; i < rows; i++) {
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