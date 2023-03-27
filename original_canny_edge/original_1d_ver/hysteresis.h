#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>

class Hysteresis
{
private:
    std::vector<double> arr;
public:
	Hysteresis();
	double* edges;
	void getHysteresis(double* image, int imgHeight, int imgWidth);
	void setupArray(double* image, int height, int width);
	int percentile(std::vector<double> vect, int percent);
	bool neighbors8(double* image, int height, int width, int x, int y);
	void deallocateVector();
};

Hysteresis::Hysteresis() {
    edges = NULL;
}

void Hysteresis::getHysteresis(double* image, int imgHeight, int imgWidth) {
    int tHi, tLo;
    bool neighbors8Bool;
    double* hysteresisImage;

    setupArray(image, imgHeight, imgWidth);

    std::sort(arr.begin(), arr.end());
    tHi = percentile(arr, 90);
    tLo = (1 / 5) * tHi;
    
    hysteresisImage = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
    for (int i = 0; i < imgHeight * imgWidth; i++) {
    	hysteresisImage[i] = image[i];
    }
    
    for (int x = 0; x < imgHeight; x++) {
        for (int y = 0; y < imgWidth; y++) {
            if (image[x * imgWidth + y] > tHi)
                hysteresisImage[x * imgWidth + y] = 255;
            else if (image[x * imgWidth + y] > tLo)
                hysteresisImage[x * imgWidth + y] = 125;
            else
                hysteresisImage[x * imgWidth + y] = 0;
        }
    }
    
    edges = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
    for (int i = 0; i < imgHeight * imgWidth; i++) {
    	this->edges[i] = hysteresisImage[i];
    }
    
    for (int x = 0; x < imgHeight; x++) {
        for (int y = 0; y < imgWidth; y++) {
            if (hysteresisImage[x * imgWidth + y] == 125) {
                neighbors8Bool = neighbors8(hysteresisImage, imgHeight, imgWidth, x, y);
                if (neighbors8Bool == true)
                    this->edges[x * imgWidth + y] = 255;
                else
                    this->edges[x * imgWidth + y] = 0;
            }
        }
    }
    
    free(hysteresisImage);
    
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

bool Hysteresis::neighbors8(double* image, int height, int width, int x, int y) {
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
