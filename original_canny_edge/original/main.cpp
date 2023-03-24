#define _USE_MATH_DEFINES
#define SIGMA 0.8

#include "pgm.h"
#include "gaussian.h"
#include "gradient.h"
#include "nonmaxSuppresion.h"
#include "hysteresis.h"
#include <chrono>

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

	gauss.gaussian(sigma);
	gauss.gaussianDeriv(sigma);
	gaussLength = gauss.getGaussianLength();

	gradient.horizontalGradient(image.matrix, gauss.g, gauss.g_deriv, imgHeight, imgWidth, gaussLength);

	gradient.verticalGradient(image.matrix, gauss.g, gauss.g_deriv, imgHeight, imgWidth, gaussLength);

	gradient.magnitudeGradient(gradient.vertical, gradient.horizontal, imgHeight, imgWidth);
	
	suppression.nonMaxSuppression(gradient.magnitude, gradient.gradient, imgHeight, imgWidth);

	hysteresis.getHysteresis(suppression.output, imgHeight, imgWidth);
	
	auto algorithm_stop = high_resolution_clock::now();
	
	writeOut(image, gradient.horizontal, "horizontalGradient.pgm", imgHeight, imgWidth);
	writeOut(image, gradient.vertical, "verticalGradient.pgm", imgHeight, imgWidth);
	writeOut(image, gradient.magnitude, "magnitudeGradient.pgm", imgHeight, imgWidth);
	writeOut(image, gradient.gradient, "iangleGradient.pgm", imgHeight, imgWidth);
	writeOut(image, suppression.output, "suppression.pgm", imgHeight, imgWidth);
	writeOut(image, hysteresis.edges, "edges.pgm", imgHeight, imgWidth);
	
	gauss.deallocateMatrix();
	gradient.deallocateMatrix(imgHeight);
	suppression.deallocateMatrix(imgHeight);
	hysteresis.deallocateMatrix(imgHeight);
	image.deallocateMatrix();
	
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
	for (int i = 0; i < imgHeight; i++)
		outMatrix[i] = new int[imgWidth];

	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			outMatrix[i][j] = (int)matrixtoWrite[i][j];
		}
	}

	image.writeImage(outName, outMatrix);

	for (int i = 0; i < imgHeight; i++)
		delete[] outMatrix[i];
	delete[] outMatrix;
	return;
}
