#include <cmath>
#include <time.h>
#include "convolve.h"

class Gradient
{
private:
	int imgHeight;
	int imgWidth;
	int gaussLength;
	int BLOCKSIZE;
public:
	Gradient();
	double* horizontal;
	double* vertical;
	double* magnitude;
	double* gradient;
	void horizontalGradient(double* image, double* gauss, double* gaussDeriv);
	void verticalGradient(double* image, double* gauss, double* gaussDeriv);
	void magnitudeGradient();
	void saveDim(int h, int w, int g, int s);
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

__global__ void cuda_mg(double* magnitude, double* gradient, double* vertical, double* horizontal, int imgHeight, int imgWidth);

void Gradient::horizontalGradient(double* image, double* gauss, double* gaussDeriv) {
	double* d_gauss;
	double* d_tempHorizontal;
	double* d_flippedGaussDeriv;
	double* d_horizontal;
	double* d_image;
	
	clock_t start, end;
	double duration;
	
	double* tempHorizontal = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	//set block dimensions
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil(imgHeight/BLOCKSIZE), ceil(imgWidth/BLOCKSIZE));
	
	start = clock();
	
	cudaMalloc((void **)&d_image, sizeof(double) * imgHeight * imgWidth);
	cudaMalloc((void **)&d_gauss, sizeof(double) * gaussLength);
	cudaMalloc((void **)&d_tempHorizontal, sizeof(double) * imgHeight * imgWidth);
	cudaMemcpy(d_image, image, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gauss, gauss, sizeof(double) * gaussLength, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cuda_convolve<<<dimGrid, dimBlock, sizeof(double) * BLOCKSIZE * BLOCKSIZE>>>(d_tempHorizontal, d_image, d_gauss, imgHeight, imgWidth, gaussLength, 1);
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
	cuda_convolve<<<dimGrid, dimBlock, sizeof(double) * BLOCKSIZE * BLOCKSIZE>>>(d_horizontal, d_tempHorizontal, d_flippedGaussDeriv, imgHeight, imgWidth, 1, gaussLength);
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
	
	//tempVertical
	double* tempVertical = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil(imgHeight/BLOCKSIZE), ceil(imgWidth/BLOCKSIZE));
	
	start = clock();
	
	cudaMalloc((void **)&d_gauss, sizeof(double) * gaussLength);
	cudaMalloc((void **)&d_image, sizeof(double) * imgHeight * imgWidth);
	cudaMalloc((void **)&d_tempVertical, sizeof(double) * imgHeight * imgWidth);
	cudaMemcpy(d_gauss, gauss, sizeof(double) * gaussLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_image, image, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cuda_convolve<<<dimGrid, dimBlock, sizeof(double) * BLOCKSIZE * BLOCKSIZE>>>(d_tempVertical, d_image, d_gauss, imgHeight, imgWidth, 1, gaussLength);
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
	cuda_convolve<<<dimGrid, dimBlock, sizeof(double) * BLOCKSIZE * BLOCKSIZE>>>(d_vertical, d_tempVertical, d_flippedGaussDeriv, imgHeight, imgWidth, gaussLength, 1);
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
	double* d_magnitude;
	double* d_gradient;
	double* d_vertical;
	double* d_horizontal;
	
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil(imgHeight/BLOCKSIZE), ceil(imgWidth/BLOCKSIZE));
	
	//magnitude
	this->magnitude = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	//gradient
	this->gradient = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	
	cudaMalloc((void **)&d_magnitude, sizeof(double)*imgHeight*imgWidth);
	cudaMalloc((void **)&d_gradient, sizeof(double)*imgHeight*imgWidth);
	cudaMalloc((void **)&d_horizontal, sizeof(double)*imgHeight*imgWidth);
	cudaMalloc((void **)&d_vertical, sizeof(double)*imgHeight*imgWidth);
	cudaMemcpy(d_horizontal, this->horizontal, sizeof(double)*imgHeight*imgWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertical, this->vertical, sizeof(double)*imgHeight*imgWidth, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cuda_mg<<<dimGrid, dimBlock>>>(d_magnitude, d_gradient, d_vertical, d_horizontal, imgHeight, imgWidth);
	cudaMemcpy(this->magnitude, d_magnitude, sizeof(double)*imgHeight*imgWidth, cudaMemcpyDeviceToHost);
	cudaMemcpy(this->gradient, d_gradient, sizeof(double)*imgHeight*imgWidth, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_magnitude);
	cudaFree(d_gradient);
	cudaFree(d_vertical);
	cudaFree(d_horizontal);
	
	return;
}

__global__ void cuda_mg(double* magnitude, double* gradient, double* vertical, double* horizontal, int imgHeight, int imgWidth) {
	int global_i = blockIdx.x * blockDim.x + threadIdx.x;
	int global_j = blockIdx.y * blockDim.y + threadIdx.y;
	double verticalSquare, horizontalSquare;
	
	verticalSquare = vertical[global_i * imgWidth + global_j] * vertical[global_i * imgWidth + global_j];
	horizontalSquare = horizontal[global_i * imgWidth + global_j] * horizontal[global_i * imgWidth + global_j];
	magnitude[global_i * imgWidth + global_j] = sqrt(verticalSquare + horizontalSquare); 
	gradient[global_i * imgWidth + global_j] = atan2(horizontal[global_i * imgWidth + global_j], vertical[global_i * imgWidth + global_j]);
	
	return;
}

void Gradient::saveDim(int h, int w, int g, int s) {
	this->imgHeight = h;
	this->imgWidth = w;
	this->gaussLength = g;
	this->BLOCKSIZE = s;
	return;
}

void Gradient::deallocateVector() {
	free(this->horizontal);
	free(this->vertical);
	free(this->magnitude);
	free(this->gradient);
	return;
}
