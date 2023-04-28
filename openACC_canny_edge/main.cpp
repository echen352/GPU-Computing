#define _USE_MATH_DEFINES
#define SIGMA 0.8

#include "pgm.h"
#include "gaussian.h"
#include "gradient.h"
#include "nonmaxSuppresion.h"
#include "hysteresis.h"
#include <openacc.h>

void writeOut(pgmImage image, Gradient gradient, nonMaxSup suppression, Hysteresis hysteresis, int imgHeight, int imgWidth);
void saveTime(double time, int size);

int main(int argc, char** argv)
{
	
	pgmImage image;
	Gauss gauss;
	Gradient gradient;
	nonMaxSup suppression;
	Hysteresis hysteresis;
	
	clock_t start, end;
	double duration;

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
	
	start = clock();
	
	gauss.gaussian(sigma);
	gauss.gaussianDeriv(sigma);
	gaussLength = gauss.getGaussianLength();
	
	gradient.saveDim(imgHeight, imgWidth, gaussLength);
	gradient.horizontalGradient(image.imgVector, gauss.g, gauss.g_deriv);
	gradient.verticalGradient(image.imgVector, gauss.g, gauss.g_deriv);
	gradient.magnitudeGradient();

	suppression.nonMaxSuppression(gradient.magnitude, gradient.gradient, imgHeight, imgWidth);

	hysteresis.getHysteresis(suppression.output, imgHeight, imgWidth);
	
	end = clock();
	duration = ((double)end - start)/CLOCKS_PER_SEC;
	printf("Algorithm Time: %f sec\n", duration);
	
	std::cout << "Saving results to file...\n";
	writeOut(image, gradient, suppression, hysteresis, imgHeight, imgWidth);
	saveTime(duration, imgWidth);
	std::cout << "Done!\n";
	
	gauss.deallocateVector();
	gradient.deallocateVector();
	suppression.deallocateVector();
	hysteresis.deallocateVector();
	image.deallocateVector();
	
	return 0;
}

void writeOut(pgmImage image, Gradient gradient, nonMaxSup suppression, Hysteresis hysteresis, int imgHeight, int imgWidth) {
	int* outVector = new int[imgHeight * imgWidth];

	for (int i = 0; i < imgHeight * imgWidth; i++)
		outVector[i] = (int)gradient.horizontal[i];
	image.writeImage("horizontalGradient.pgm", outVector);

	for (int i = 0; i < imgHeight * imgWidth; i++)
		outVector[i] = (int)gradient.vertical[i];
	image.writeImage("verticalGradient.pgm", outVector);
	
	for (int i = 0; i < imgHeight * imgWidth; i++)
		outVector[i] = (int)gradient.magnitude[i];
	image.writeImage("magnitudeGradient.pgm", outVector);
	
	for (int i = 0; i < imgHeight * imgWidth; i++)
		outVector[i] = (int)gradient.gradient[i];
	image.writeImage("iangleGradient.pgm", outVector);

	for (int i = 0; i < imgHeight * imgWidth; i++)
		outVector[i] = (int)suppression.output[i];
	image.writeImage("suppression.pgm", outVector);

	for (int i = 0; i < imgHeight * imgWidth; i++)
		outVector[i] = (int)hysteresis.edges[i];
	image.writeImage("edges.pgm", outVector);

	delete[] outVector;
	return;
}

void saveTime(double time, int size) {
	FILE* f;
	f = fopen("openACC_timings.csv", "a+");
	fprintf(f, "%d,%f\n", size, time);
	fclose(f);
	return;
}
