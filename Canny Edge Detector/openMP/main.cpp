#define _USE_MATH_DEFINES
#define NUMTHREADS 16
#define SIGMA 0.8

#include "pgm.h"
#include "gaussian.h"
#include "gradient.h"
#include "nonmaxSuppresion.h"
#include "hysteresis.h"
#include <fstream>
#include <string>
#include <chrono>
#include <omp.h>

using namespace std::chrono;

void writeOut(pgmImage image, double** matrixtoWrite, const char* outName, int imgHeight, int imgWidth);
void saveTime(int algorithm_duration, int imgHeight);

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
	int imgHeight, imgWidth, gaussLength, algoTime;

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
	
	std::cout << "Number of Processors: " << omp_get_num_procs() << std::endl;
	std::cout << "Attempting to Set Num Threads: " << NUMTHREADS << std::endl;
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
	algoTime = algorithm_duration.count();
	std::cout << "Time taken by canny edge detector algorithm: " << algoTime << " us" << std::endl;
	
	auto program_duration = duration_cast<microseconds>(program_stop - program_start);
	std::cout << "Total time taken by program: " << program_duration.count() << " us" << std::endl;
	
	saveTime(algoTime, imgHeight);
	
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

void saveTime(int time, int size) {
	std::ofstream csvFile;
	std::string Str;
	csvFile.open("timings.csv", std::ios::app);
	Str = std::to_string(size) + "," + std::to_string(NUMTHREADS) + "," + std::to_string(time) + "\n";
	csvFile << Str;
	csvFile.close();
	return;
}
