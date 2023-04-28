#include <vector>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdlib>

class Hysteresis
{
public:
	Hysteresis();
	double* edges;
	void getHysteresis(double* image, int imgHeight, int imgWidth, int BLOCKSIZE);
	int percentile(double** arr, int percent, int height, int width);
	void deallocateVector();
};

Hysteresis::Hysteresis() {
    edges = NULL;
}

__global__ void getHysteresisImage(double* hysteresisImage, double* image, int height, int width, int tHi, int tLo);
__global__ void getEdges(double* edges, double* hysteresisImage, int imgHeight, int imgWidth);
__device__ bool neighbors8(double* image, int height, int width, int x, int y);

void Hysteresis::getHysteresis(double* image, int imgHeight, int imgWidth, int BLOCKSIZE) {
    int tHi, tLo;
    double* arr;
    double* hysteresisImage;

    clock_t start, end;
    double duration;
    
    start = clock();
    
    arr = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
    memcpy(arr, image, sizeof(double)*imgHeight*imgWidth);
    double* d_arr;
    
    cudaMalloc((void **)&d_arr, sizeof(double)*imgHeight*imgWidth);
    cudaMemcpy(d_arr, arr, sizeof(double)*imgHeight*imgWidth, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    thrust::device_ptr<double> ptr(d_arr);
    thrust::sort(ptr, ptr + imgHeight*imgWidth);
    cudaMemcpy(arr, d_arr, sizeof(double)*imgHeight*imgWidth, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_arr);
    tHi = percentile(&arr, 90, imgHeight, imgWidth);
    if (tHi < 0) {printf("Error Calculating n percentile in Hystersis!");}
    tLo = (1 / 5) * tHi;
    
    hysteresisImage = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
    memcpy(hysteresisImage, image, sizeof(double)*imgHeight*imgWidth);
		
	//set block dimensions
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil(imgHeight/BLOCKSIZE), ceil(imgWidth/BLOCKSIZE));
	double* d_hysteresisImage;
	double* d_image;
	double* d_edges;
	
	cudaMalloc((void **)&d_image, sizeof(double) * imgHeight * imgWidth);
	cudaMalloc((void **)&d_hysteresisImage, sizeof(double) * imgHeight * imgWidth);
	cudaMemcpy(d_image, image, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hysteresisImage, image, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	getHysteresisImage<<<dimGrid, dimBlock>>>(d_hysteresisImage, d_image, imgHeight, imgWidth, tHi, tLo);
	cudaMemcpy(hysteresisImage, d_hysteresisImage, sizeof(double) * imgHeight * imgWidth, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	end = clock();
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("Hysteresis: %f sec\n", duration);
	
	cudaFree(d_hysteresisImage);
	cudaFree(d_image);
	
	start = clock(); 
	  
    edges = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
    memcpy(edges, hysteresisImage, sizeof(double)*imgHeight*imgWidth);
		
    cudaMalloc((void **)&d_edges, sizeof(double) * imgHeight * imgWidth);
    cudaMalloc((void **)&d_hysteresisImage, sizeof(double) * imgHeight * imgWidth);
    cudaMemcpy(d_edges, edges, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hysteresisImage, hysteresisImage, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    getEdges<<<dimGrid, dimBlock>>>(d_edges, d_hysteresisImage, imgHeight, imgWidth);
    cudaMemcpy(edges, d_edges, sizeof(double) * imgHeight * imgWidth, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    end = clock();
    duration = ((double)end - start)/CLOCKS_PER_SEC;
    printf("Edge Linking: %f sec\n", duration);
    
    cudaFree(d_hysteresisImage);
    cudaFree(d_edges);
    
    free(arr);
    free(hysteresisImage);
    
    return;
}

__global__ void getHysteresisImage(double* hysteresisImage, double* image, int imgHeight, int imgWidth, int tHi, int tLo) {
	int global_i = blockIdx.x * blockDim.x + threadIdx.x;
	int global_j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (image[global_i * imgWidth + global_j] > tHi)
		hysteresisImage[global_i * imgWidth + global_j] = 255;
	else if (image[global_i * imgWidth + global_j] > tLo)
		hysteresisImage[global_i * imgWidth + global_j] = 125;
	else
		hysteresisImage[global_i * imgWidth + global_j] = 0;
		
	return;
}

__global__ void getEdges(double* edges, double* hysteresisImage, int imgHeight, int imgWidth) {
    bool neighbors8Bool;
    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    int global_j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (hysteresisImage[global_i * imgWidth + global_j] == 125) {
        neighbors8Bool = neighbors8(hysteresisImage, imgHeight, imgWidth, global_i, global_j);
        if (neighbors8Bool == true)
            edges[global_i * imgWidth + global_j] = 255;
        else
            edges[global_i * imgWidth + global_j] = 0;
    }
    
    return;
}

int Hysteresis::percentile(double** arr, int percent, int height, int width) {
    int n = height * width;
    double p;
    
    for (int i = n - 1; i > -1; i--) {
        p = 100 * (i + 0.5) / n;
        if (ceil(p) == percent)
        	return (*arr)[i];
    }

    return -1;
}

__device__ bool neighbors8(double* image, int height, int width, int x, int y) {
    if (x - 1 < 1 || x + 1 > height || y - 1 < 1 || y + 1 > width)
        return false;

    if (image[(x - 1) * width + y] == 255)
        return true;
    else if (image[(x - 1) * width + (y + 1)] == 255)
        return true;
    else if (image[x * width + (y + 1)] == 255)
        return true;
    else if (image[(x + 1) * width + (y + 1)] == 255)
        return true;
    else if (image[(x + 1) * width + y] == 255)
        return true;
    else if (image[(x + 1) * width + (y - 1)] == 255)
        return true;
    else if (image[x * width + (y - 1)] == 255)
        return true;
    else if (image[(x - 1) * width + (y - 1)] == 255)
        return true;
    else
        return false;
}

void Hysteresis::deallocateVector() {
    free(this->edges);
    return;
}
