#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <cuda.h>
//#include <cuda_runtime.h>
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
	double* d_gauss;
	double* d_tempHorizontal;
	double* d_flippedGaussDeriv;
	double* d_horizontal;
	double* d_image;
	
	clock_t start, end;
	double duration;
	
	//set block dimensions
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid((imgHeight+BLOCKSIZE)/BLOCKSIZE, (imgWidth+BLOCKSIZE)/BLOCKSIZE);
	
	double* tempHorizontal = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	start = clock();
	
	cudaMalloc((void **)&d_image, sizeof(double) * imgHeight * imgWidth);
	cudaMalloc((void **)&d_gauss, sizeof(double) * gaussLength);
	cudaMalloc((void **)&d_tempHorizontal, sizeof(double) * imgHeight * imgWidth);
	cudaMemcpy(d_image, image, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gauss, gauss, sizeof(double) * gaussLength, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	//convolve_1d(&tempHorizontal, &image, &gauss, imgHeight, imgWidth, gaussLength, 1);
	cuda_convolve<<<dimGrid, dimBlock>>>(d_tempHorizontal, d_image, d_gauss, imgHeight, imgWidth, gaussLength, 1);
	cudaMemcpy(tempHorizontal, d_tempHorizontal, sizeof(double) * imgHeight * imgWidth, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	end = clock();
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("Temp Horizontal Convolution: %f sec\n", duration);
	
	cudaFree(d_gauss);
	cudaFree(d_image);
	cudaFree(d_tempHorizontal);
	
	//flip gaussian deriv mask
	double* flippedGaussDeriv = (double*)malloc(sizeof(double)*gaussLength);
	for (int i = 0; i < gaussLength; i++)
		flippedGaussDeriv[i] = gaussDeriv[i] * -1;
	
	this->horizontal = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	start = clock();
	
	cudaMalloc((void **)&d_flippedGaussDeriv, sizeof(double) * gaussLength);
	cudaMalloc((void **)&d_tempHorizontal, sizeof(double) * imgHeight * imgWidth);
	cudaMalloc((void **)&d_horizontal, sizeof(double) * imgHeight * imgWidth);
	cudaMemcpy(d_flippedGaussDeriv, flippedGaussDeriv, sizeof(double) * gaussLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tempHorizontal, tempHorizontal, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	//convolve_1d(&horizontal, &tempHorizontal, &flippedGaussDeriv, imgHeight, imgWidth, 1, gaussLength);
	cuda_convolve<<<dimGrid, dimBlock>>>(d_horizontal, d_tempHorizontal, d_flippedGaussDeriv, imgHeight, imgWidth, 1, gaussLength);
	cudaMemcpy(horizontal, d_horizontal, sizeof(double) * imgHeight * imgWidth, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	end = clock();
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("Horizontal Convolution: %f sec\n", duration);
	
	cudaFree(d_flippedGaussDeriv);
	cudaFree(d_tempHorizontal);
	cudaFree(d_horizontal);
	
	free(tempHorizontal);
	free(flippedGaussDeriv);
	
	return;
}

void Gradient::verticalGradient(double* image, double* gauss, double* gaussDeriv) {
	double* d_tempVertical;
	double* d_image;
	double* d_gauss;
	double* d_vertical;
	double* d_flippedGaussDeriv;
	
	clock_t start, end;
	double duration;
	
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid((imgHeight+BLOCKSIZE)/BLOCKSIZE, (imgWidth+BLOCKSIZE)/BLOCKSIZE);

	//tempVertical
	double* tempVertical = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	start = clock();
	
	cudaMalloc((void **)&d_gauss, sizeof(double) * gaussLength);
	cudaMalloc((void **)&d_image, sizeof(double) * imgHeight * imgWidth);
	cudaMalloc((void **)&d_tempVertical, sizeof(double) * imgHeight * imgWidth);
	cudaMemcpy(d_gauss, gauss, sizeof(double) * gaussLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_image, image, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	//convolve_1d(&tempVertical, &image, &gauss, imgHeight, imgWidth, 1, gaussLength);
	cuda_convolve<<<dimGrid, dimBlock>>>(d_tempVertical, d_image, d_gauss, imgHeight, imgWidth, 1, gaussLength);
	cudaMemcpy(tempVertical, d_tempVertical, sizeof(double) * imgHeight * imgWidth, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	end = clock();
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("Temp Vertical Convolution: %f sec\n", duration);
	
	cudaFree(d_tempVertical);
	cudaFree(d_image);
	cudaFree(d_gauss);
	
	//flippedGaussDeriv
	double* flippedGaussDeriv = (double*)malloc(sizeof(double)*gaussLength);
	for (int i = 0; i < gaussLength; i++)
		flippedGaussDeriv[i] =gaussDeriv[i] * -1;
	
	//vertical
	this->vertical = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	start = clock();
	
	cudaMalloc((void **)&d_flippedGaussDeriv, sizeof(double) * gaussLength);
	cudaMalloc((void **)&d_tempVertical, sizeof(double) * imgHeight * imgWidth);
	cudaMalloc((void **)&d_vertical, sizeof(double) * imgHeight * imgWidth);
	cudaMemcpy(d_flippedGaussDeriv, flippedGaussDeriv, sizeof(double) * gaussLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tempVertical, tempVertical, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	//convolve_1d(&vertical, &tempVertical, &flippedGaussDeriv, imgHeight, imgWidth, gaussLength, 1);
	cuda_convolve<<<dimGrid, dimBlock>>>(d_vertical, d_tempVertical, d_flippedGaussDeriv, imgHeight, imgWidth, gaussLength, 1);
	cudaMemcpy(vertical, d_vertical, sizeof(double) * imgHeight * imgWidth, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	end = clock();
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("Vertical Convolution: %f sec\n", duration);
	
	cudaFree(d_vertical);
	cudaFree(d_tempVertical);
	cudaFree(d_flippedGaussDeriv);
	
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
