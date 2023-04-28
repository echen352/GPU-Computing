#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include "convolve.h"

class Gradient
{
private:
	int imgHeight;
	int imgWidth;
	int gaussLength;
public:
	Gradient();
	double* horizontal;
	double* vertical;
	double* magnitude;
	double* gradient;
	void horizontalGradient(double* image, double* gauss, double* gaussDeriv);
	void verticalGradient(double* image, double* gauss, double* gaussDeriv);
	void magnitudeGradient();
	void saveDim(int h, int w, int g);
	void deallocateVector();
};

Gradient::Gradient() {
	imgHeight = 0;
	imgWidth = 0;
	gaussLength = 0;
	horizontal = NULL;
	vertical = NULL;
	magnitude = NULL;
	gradient = NULL;
}

void Gradient::horizontalGradient(double* image, double* gauss, double* gaussDeriv) {
	clock_t start, end;
	double duration;
	
	double* tempHorizontal = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	start = clock();
	
	convolve_1d(tempHorizontal, image, gauss, imgHeight, imgWidth, gaussLength, 1);
	
	end = clock();
	
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("Temp Horizontal Convolution: %f sec\n", duration);
	
	//flip gaussian deriv mask
	double* flippedGaussDeriv = (double*)malloc(sizeof(double)*gaussLength);
	for (int i = 0; i < gaussLength; i++)
		flippedGaussDeriv[i] = gaussDeriv[i] * -1;
	
	this->horizontal = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	start = clock();
	
	convolve_1d(horizontal, tempHorizontal, flippedGaussDeriv, imgHeight, imgWidth, 1, gaussLength);
	
	end = clock();
	
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("Horizontal Convolution: %f sec\n", duration);
	
	free(tempHorizontal);
	free(flippedGaussDeriv);
	
	return;
}

void Gradient::verticalGradient(double* image, double* gauss, double* gaussDeriv) {
	clock_t start, end;
	double duration;

	//tempVertical
	double* tempVertical = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	start = clock();
	
	convolve_1d(tempVertical, image, gauss, imgHeight, imgWidth, 1, gaussLength);
	
	end = clock();
	
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("Temp Vertical Convolution: %f sec\n", duration);
	
	//flippedGaussDeriv
	double* flippedGaussDeriv = (double*)malloc(sizeof(double)*gaussLength);
	for (int i = 0; i < gaussLength; i++)
		flippedGaussDeriv[i] =gaussDeriv[i] * -1;
	
	//vertical
	this->vertical = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	start = clock();
	
	convolve_1d(vertical, tempVertical, flippedGaussDeriv, imgHeight, imgWidth, gaussLength, 1);
	
	end = clock();
	
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("Vertical Convolution: %f sec\n", duration);
	
	free(tempVertical);
	free(flippedGaussDeriv);
	
	return;
}

void Gradient::magnitudeGradient() {
	double verticalSquare;
	double horizontalSquare;
	
	//magnitude
	this->magnitude = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	//gradient
	this->gradient = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	#pragma acc parallel loop
	for (int x = 0; x < imgHeight; x++) {
		for (int y = 0; y < imgWidth; y++) {
			verticalSquare = this->vertical[x * imgWidth + y] * this->vertical[x * imgWidth + y];
			horizontalSquare = this->horizontal[x * imgWidth + y] * this->horizontal[x * imgWidth + y];
			this->magnitude[x * imgWidth + y] = sqrt(verticalSquare + horizontalSquare); 
			this->gradient[x * imgWidth + y] = atan2(this->horizontal[x * imgWidth + y], this->vertical[x * imgWidth + y]);
		}
	}
	
	return;
}

void Gradient::saveDim(int h, int w, int g) {
	this->imgHeight = h;
	this->imgWidth = w;
	this->gaussLength = g;
	return;
}

void Gradient::deallocateVector() {
	free(this->horizontal);
	free(this->vertical);
	free(this->magnitude);
	free(this->gradient);
	return;
}
