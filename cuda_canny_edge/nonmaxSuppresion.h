#include <cmath>
#include <stdio.h>

class nonMaxSup
{
public:
	nonMaxSup();
	double* output;
	void runNonMaxSup(double* gxy, double* iangle, int gxyHeight, int gxyWidth, int BLOCKSIZE);
	void deallocateVector();
};

nonMaxSup::nonMaxSup() {
	output = NULL;
}

__global__ void cuda_nonMaxSuppression(double* output, double* gxy, double* iangle, int gxyHeight, int gxyWidth);

void nonMaxSup::runNonMaxSup(double* gxy, double* iangle, int gxyHeight, int gxyWidth, int BLOCKSIZE) {
	clock_t start, end;
	double duration;

	this->output = (double*)malloc(sizeof(double) * gxyHeight * gxyWidth);
	memcpy(this->output, gxy, sizeof(double) * gxyHeight * gxyWidth);
	
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil(gxyHeight/BLOCKSIZE), ceil(gxyWidth/BLOCKSIZE));
	double* d_gxy;
	double* d_iangle;
	double* d_output;
	
	start = clock();
	
	cudaMalloc((void **)&d_gxy, sizeof(double) * gxyHeight * gxyWidth);
	cudaMalloc((void **)&d_iangle, sizeof(double) * gxyHeight * gxyWidth);
	cudaMalloc((void **)&d_output, sizeof(double) * gxyHeight * gxyWidth);
	cudaMemcpy(d_gxy, gxy, sizeof(double) * gxyHeight * gxyWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(d_iangle, iangle, sizeof(double) * gxyHeight * gxyWidth, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, output, sizeof(double) * gxyHeight * gxyWidth, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cuda_nonMaxSuppression<<<dimGrid, dimBlock>>>(d_output, d_gxy, d_iangle, gxyHeight, gxyWidth);
	cudaMemcpy(this->output, d_output, sizeof(double) * gxyHeight * gxyWidth, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	end = clock();
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("NonMax Suppression: %f sec\n", duration);
	
	cudaFree(d_gxy);
	cudaFree(d_iangle);
	cudaFree(d_output);
	
	return;
}

__global__ void cuda_nonMaxSuppression(double* output, double* gxy, double* iangle, int gxyHeight, int gxyWidth) {
	double theta;
	double center;
	int global_i, global_j;
	
	// map global IDs
	global_i = blockIdx.x * blockDim.x + threadIdx.x;
	global_j = blockIdx.y * blockDim.y + threadIdx.y;
	
		theta = iangle[global_i * gxyWidth + global_j];
		if (theta < 0)
			theta += M_PI;
		theta = theta * (180 / M_PI);
		if (global_i - 1 > -1 && global_i + 1 < gxyHeight && global_j - 1 > -1 && global_j + 1 < gxyWidth) {
			center = gxy[global_i * gxyWidth + global_j];
			if (theta <= 22.5 || theta > 157.5) {
				if (center < gxy[(global_i - 1) * gxyWidth + global_j] || center < gxy[(global_i + 1) * gxyWidth + global_j])
					output[global_i * gxyWidth + global_j] = 0;
			}
			else if (theta > 22.5 && theta <= 67.5) {
				if (center < gxy[(global_i - 1) * gxyWidth + (global_j - 1)] || center < gxy[(global_i + 1) * gxyWidth + (global_j + 1)])
					output[global_i * gxyWidth + global_j] = 0;
			}
			else if (theta > 67.5 && theta <= 112.5) {
				if (center < gxy[global_i * gxyWidth + (global_j - 1)] || center < gxy[global_i * gxyWidth + (global_j + 1)])
					output[global_i * gxyWidth + global_j] = 0;
			}
			else if (theta > 112.5 && theta <= 157.5) {
				if (center < gxy[(global_i + 1) * gxyWidth + (global_j - 1)] || center < gxy[(global_i - 1) * gxyWidth + (global_j + 1)])
					output[global_i * gxyWidth + global_j] = 0;
			}
		}

	return;
}

void nonMaxSup::deallocateVector() {
    free(this->output);
    return;
}
