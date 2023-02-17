#define _USE_MATH_DEFINES
#define SIGMA 0.8

#include "pgm.h"
#include "gaussian.h"
#include "gradient.h"
#include "nonmaxSuppresion.h"
#include "hysteresis.h"
#include <chrono>
#include <omp.h>

using namespace std::chrono;

void writeOut(pgmImage image, double** matrixtoWrite, const char* outName, int imgHeight, int imgWidth);

int main(int argc, char** argv)
{
	auto program_start = high_resolution_clock::now();
	
	pgmImage image;
	Gauss gauss;
	Gradient gradient;
	nonMaxSup suppression;
	Hysteresis hysteresis;

	const char* imageName;
	double sigma = SIGMA;
	int imgHeight, imgWidth, gaussLength;

	if (argc == 2) {
		imageName = argv[1];
	} else {
		std::cout << "Not enough inputs: expected 2, received " << argc << std::endl;
		return -1;
	}

	std::cout << "In File Name: " << imageName << std::endl;
	image.readImage(imageName);
	imgHeight = image.getHeight();
	imgWidth = image.getWidth();
	
	auto algorithm_start = high_resolution_clock::now();
	
	#pragma omp parallel default(shared)
	{
		#pragma omp single
		{
			#pragma omp task
				gauss.gaussian(sigma);
			#pragma omp task
				gauss.gaussianDeriv(sigma);
		}
	}
	
	gaussLength = gauss.getGaussianLength();
	
	#pragma omp parallel default(shared)
	{
		#pragma omp single
		{
			#pragma omp task
				gradient.horizontalGradient(image.matrix, gauss.g, gauss.g_deriv, imgHeight, imgWidth, gaussLength);
			#pragma omp task
				gradient.verticalGradient(image.matrix, gauss.g, gauss.g_deriv, imgHeight, imgWidth, gaussLength);
		}
	}
	
	gradient.magnitudeGradient(gradient.vertical, gradient.horizontal, imgHeight, imgWidth);

	suppression.nonMaxSuppression(gradient.magnitude, gradient.gradient, imgHeight, imgWidth);

	hysteresis.getHysteresis(suppression.output, imgHeight, imgWidth);
	
	auto algorithm_stop = high_resolution_clock::now();
	
	#pragma omp parallel default(shared)
	{
		#pragma omp single
		{
			#pragma omp task
				writeOut(image, gradient.horizontal, "horizontalGradient.pgm", imgHeight, imgWidth);
			#pragma omp task
				writeOut(image, gradient.vertical, "verticalGradient.pgm", imgHeight, imgWidth);
			#pragma omp task
				writeOut(image, gradient.magnitude, "magnitudeGradient.pgm", imgHeight, imgWidth);
			#pragma omp task
				writeOut(image, gradient.gradient, "iangleGradient.pgm", imgHeight, imgWidth);
			#pragma omp task
				writeOut(image, suppression.output, "suppression.pgm", imgHeight, imgWidth);
			#pragma omp task
				writeOut(image, hysteresis.edges, "edges.pgm", imgHeight, imgWidth);
		}
	}
	
	#pragma omp parallel default(shared)
	{
		#pragma omp single
		{
			#pragma omp task
				gauss.deallocateMatrix();
			#pragma omp task
				gradient.deallocateMatrix(imgHeight);
			#pragma omp task
				suppression.deallocateMatrix(imgHeight);
			#pragma omp task
				hysteresis.deallocateMatrix(imgHeight);
			#pragma omp task
				image.deallocateMatrix();
		}
	}
	
	auto program_stop = high_resolution_clock::now();
	
	auto algorithm_duration = duration_cast<microseconds>(algorithm_stop - algorithm_start);
	std::cout << "Time taken by canny edge detector algorithm: " << algorithm_duration.count() << " us" << std::endl;
	
	auto program_duration = duration_cast<microseconds>(program_stop - program_start);
	std::cout << "Total time taken by program: " << program_duration.count() << " us" << std::endl;
	return 0;
}

void writeOut(pgmImage image, double** matrixtoWrite, const char* outName, int imgHeight, int imgWidth) {
	int** outMatrix;
	outMatrix = new int* [imgHeight];
	#pragma omp parallel for
		for (int i = 0; i < imgHeight; i++)
			outMatrix[i] = new int[imgWidth];
	#pragma omp parallel for collapse(2)
		for (int i = 0; i < imgHeight; i++) {
			for (int j = 0; j < imgWidth; j++) {
				outMatrix[i][j] = (int)matrixtoWrite[i][j];
			}
		}

	image.writeImage(outName, outMatrix);
	#pragma omp parallel for
		for (int i = 0; i < imgHeight; i++)
			delete[] outMatrix[i];
	delete[] outMatrix;
	return;
}
