#include <vector>
#include <algorithm>
#include <cmath>
#include <stdio.h>

class Hysteresis
{
private:
    std::vector<double> arr;
public:
	Hysteresis();
	double* edges;
	void getHysteresis(double* image, int imgHeight, int imgWidth, int BLOCKSIZE);
	void setupArray(double* image, int height, int width);
	int percentile(std::vector<double> vect, int percent);
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
    double* hysteresisImage;

    clock_t start, end;
    double duration;
    
    start = clock();
    
    setupArray(image, imgHeight, imgWidth);

    std::sort(arr.begin(), arr.end());
    tHi = percentile(arr, 90);
    tLo = (1 / 5) * tHi;
    
    hysteresisImage = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
	for (int i = 0; i < imgHeight * imgWidth; i++)
		hysteresisImage[i] = image[i];
		
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
	for (int i = 0; i < imgHeight * imgWidth; i++)
		this->edges[i] = hysteresisImage[i];
		
    cudaMalloc((void **)&d_edges, sizeof(double) * imgHeight * imgWidth);
    cudaMalloc((void **)&d_hysteresisImage, sizeof(double) * imgHeight * imgWidth);
    cudaMemcpy(d_edges, this->edges, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hysteresisImage, hysteresisImage, sizeof(double) * imgHeight * imgWidth, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    getEdges<<<dimGrid, dimBlock>>>(d_edges, d_hysteresisImage, imgHeight, imgWidth);
    cudaMemcpy(this->edges, d_edges, sizeof(double) * imgHeight * imgWidth, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    end = clock();
    duration = ((double)end - start)/CLOCKS_PER_SEC;
    printf("Edge Linking: %f sec\n", duration);
    
    cudaFree(d_hysteresisImage);
    cudaFree(d_edges);
    
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

void Hysteresis::setupArray(double* image, int height, int width) {
    double value;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            value = image[i * width + j];
            this->arr.push_back(value);
        }
    }

    return;
}

int Hysteresis::percentile(std::vector<double> vect, int percent) {
    std::vector<double> prcVector;
    int n = vect.size();
    double p;
    for (int i = 0; i < n; i++) {
        p = 100 * (i + 0.5) / n;
        prcVector.push_back(p);
    }

    for (int i = 0; i < n; i++) {
        if (floor(prcVector[i]) == percent)
            return vect[i];
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
