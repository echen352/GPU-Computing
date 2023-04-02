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
	int percentile(double** arr, int percent, int height, int width);
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
    double* arr;
    
    clock_t start, end;
    double duration;
    
    start = clock();
    
    arr = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
    memcpy(arr, image, sizeof(double)*imgHeight*imgWidth);

    std::sort(arr, arr + imgHeight*imgWidth);
    tHi = percentile(&arr, 90, imgHeight, imgWidth);
    if (tHi < 0) {printf("Error Calculating n percentile in Hystersis!");}
    tLo = (1 / 5) * tHi;
    
    hysteresisImage = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
    memcpy(hysteresisImage, image, sizeof(double)*imgHeight*imgWidth);
    
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
    
    end = clock();
    duration = ((double)end - start)/CLOCKS_PER_SEC;
    printf("Hysteresis: %f sec\n", duration);
    
    start = clock();   
    
    edges = (double*)malloc(sizeof(double)*imgHeight*imgWidth);
    memcpy(this->edges, hysteresisImage, sizeof(double)*imgHeight*imgWidth);
    
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
    
    end = clock();
    duration = ((double)end - start)/CLOCKS_PER_SEC;
    printf("Edge Linking: %f sec\n", duration);
    
    free(arr);
    free(hysteresisImage);
    
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
