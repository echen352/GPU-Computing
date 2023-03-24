#include <iostream>
#include <vector>
#include <algorithm>

class Hysteresis
{
private:
	std::vector<double> arr;
public:
	Hysteresis();
	double** edges;
	void getHysteresis(double** image, int imgHeight, int imgWidth);
	void setupArray(double** image, int height, int width);
	double** copyMatrix(double** image, int height, int width);
	int percentile(std::vector<double> vect, int percent);
	bool neighbors8(double** image, int height, int width, int x, int y);
	void deallocateMatrix(int rows);
};

Hysteresis::Hysteresis() {
	edges = NULL;
}

void Hysteresis::getHysteresis(double** image, int imgHeight, int imgWidth) {
	int tHi, tLo, i, x, y;
	bool neighbors8Bool;
	double** hysteresisImage;

	setupArray(image, imgHeight, imgWidth);

	std::sort(arr.begin(), arr.end());
	tHi = percentile(arr, 90);
	tLo = (1 / 5) * tHi;
	
	hysteresisImage = copyMatrix(image, imgHeight, imgWidth);
	
	#pragma omp parallel for private(y) num_threads(NUMTHREADS)
	for (x = 0; x < imgHeight; x++) {
	    for (y = 0; y < imgWidth; y++) {
		if (image[x][y] > tHi)
		    hysteresisImage[x][y] = 255;
		else if (image[x][y] > tLo)
		    hysteresisImage[x][y] = 125;
		else
		    hysteresisImage[x][y] = 0;
	    }
	}

	this->edges = copyMatrix(hysteresisImage, imgHeight, imgWidth);
	
	#pragma omp parallel for private(y, neighbors8Bool) num_threads(NUMTHREADS)
	for (x = 0; x < imgHeight; x++) {
	    for (y = 0; y < imgWidth; y++) {
		if (hysteresisImage[x][y] == 125) {
		    neighbors8Bool = neighbors8(hysteresisImage, imgHeight, imgWidth, x, y);
		    if (neighbors8Bool == true)
			this->edges[x][y] = 255;
		    else
			this->edges[x][y] = 0;
		}
	    }
	}

	for (i = 0; i < imgHeight; i++)
		free(hysteresisImage[i]);
	free(hysteresisImage);

	return;
}

void Hysteresis::setupArray(double** image, int height, int width) {
	double value;
	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			value = image[i][j];
			this->arr.push_back(value);
		}
	}

	return;
}

double** Hysteresis::copyMatrix(double** image, int height, int width) {
	double** newMatrix;
	int i, j;

	newMatrix = (double**)malloc(sizeof(double*) * height);
	if (newMatrix == NULL) {
		std::cout << "Error allocating memory" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < height; i++) {
	    newMatrix[i] = (double*)malloc(sizeof(double) * width);
	    if (newMatrix[i] == NULL) {
		std::cout << "Error allocating memory" << std::endl;
		exit(EXIT_FAILURE);
	    }
	}
	
	#pragma omp parallel for private(j) num_threads(NUMTHREADS)
	for (i = 0; i < height; i++) {
	    for (j = 0; j < width; j++)
		newMatrix[i][j] = image[i][j];
	}
			
	return newMatrix;
}

int Hysteresis::percentile(std::vector<double> vect, int percent) {
	std::vector<double> prcVector;
	int n = vect.size();
	int i;
	double p;
	double r;

	for (i = 0; i < n; i++) {
	    p = 100 * (i + 0.5) / n;
	    prcVector.push_back(p);
	}

	for (i = 0; i < n; i++) {
		if (floor(prcVector[i]) == percent)
			r = vect[i];
	}

	return r;
}

bool Hysteresis::neighbors8(double** image, int height, int width, int x, int y) {
	if (x - 1 < 1 || x + 1 > height || y - 1 < 1 || y + 1 > width)
		return false;
	if (image[x - 1][y] == 255)
		return true;
	else if (image[x - 1][y + 1] == 255)
		return true;
	else if (image[x][y + 1] == 255)
		return true;
	else if (image[x + 1][y + 1] == 255)
		return true;
	else if (image[x + 1][y] == 255)
		return true;
	else if (image[x + 1][y - 1] == 255)
		return true;
	else if (image[x][y - 1] == 255)
		return true;
	else if (image[x - 1][y - 1] == 255)
		return true;
	else
		return false;
}

void Hysteresis::deallocateMatrix(int rows) {
	int i;

	for (i = 0; i < rows; i++)
		free(this->edges[i]);
	free(this->edges);

	return;
}
