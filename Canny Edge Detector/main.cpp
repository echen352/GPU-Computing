#define _USE_MATH_DEFINES

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "pgm.h"
#include "gaussian.h"
#include "gradient.h"
#include "nonmaxSuppresion.h"
#include "hysteresis.h"

#define SIGMA 0.8

void writeOut(pgmImage image, double** matrixtoWrite, const char* outName, int imgHeight, int imgWidth);

int main(int argc, char** argv)
{
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

	gauss.gaussian(sigma);
	gauss.gaussianDeriv(sigma);
	gaussLength = gauss.getGaussianLength();

	gradient.horizontalGradient(image.matrix, gauss.g, gauss.g_deriv, imgHeight, imgWidth, gaussLength);
	writeOut(image, gradient.horizontal, "C:\\Users\\ellis\\Desktop\\horizontalGradient.pgm", imgHeight, imgWidth);

	gradient.verticalGradient(image.matrix, gauss.g, gauss.g_deriv, imgHeight, imgWidth, gaussLength);
	writeOut(image, gradient.vertical, "C:\\Users\\ellis\\Desktop\\verticalGradient.pgm", imgHeight, imgWidth);

	gradient.magnitudeGradient(gradient.vertical, gradient.horizontal, imgHeight, imgWidth);
	writeOut(image, gradient.magnitude, "C:\\Users\\ellis\\Desktop\\magnitudeGradient.pgm", imgHeight, imgWidth);
	writeOut(image, gradient.gradient, "C:\\Users\\ellis\\Desktop\\iangleGradient.pgm", imgHeight, imgWidth);

	suppression.nonMaxSuppression(gradient.magnitude, gradient.gradient, imgHeight, imgWidth);
	writeOut(image, suppression.output, "C:\\Users\\ellis\\Desktop\\suppression.pgm", imgHeight, imgWidth);

	hysteresis.getHysteresis(suppression.output, imgHeight, imgWidth);
	writeOut(image, hysteresis.edges, "C:\\Users\\ellis\\Desktop\\edges.pgm", imgHeight, imgWidth);

	/*cv::Mat greyImage = cv::Mat(imgHeight, imgWidth, CV_8U, hysteresis.edges);
	cv::namedWindow("Edges", cv::WINDOW_AUTOSIZE);
	cv::imshow("Edges", greyImage);
	cv::waitKey(0);
	cv::destroyAllWindows();*/

	gauss.deallocateMatrix();
	gradient.deallocateMatrix(imgHeight);
	suppression.deallocateMatrix(imgHeight);
	hysteresis.deallocateMatrix(imgHeight);
	image.deallocateMatrix();
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