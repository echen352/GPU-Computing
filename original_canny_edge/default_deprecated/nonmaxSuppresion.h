#include <iostream>
#include <cmath>

class nonMaxSup
{
public:
	nonMaxSup();
	double** output;
	void nonMaxSuppression(double** gxy, double** iangle, int gxyHeight, int gxyWidth);
	double** copyGXYtoOutput(double** gxy, int height, int width);
    void deallocateMatrix(int rows);
};

nonMaxSup::nonMaxSup() {
	output = NULL;
}

void nonMaxSup::nonMaxSuppression(double** gxy, double** iangle, int gxyHeight, int gxyWidth) {
    double theta;
    double center;

    this->output = copyGXYtoOutput(gxy, gxyHeight, gxyWidth);

    for (int x = 0; x < gxyHeight; x++) {
        for (int y = 0; y < gxyWidth; y++) {
            theta = iangle[x][y];
            if (theta < 0)
                theta += M_PI;
            theta = theta * (180 / M_PI);
            if (x - 1 > -1 && x + 1 < gxyHeight && y - 1 > -1 && y + 1 < gxyWidth) {
                center = gxy[x][y];
                if (theta <= 22.5 || theta > 157.5) {
                    if (center < gxy[x - 1][y] || center < gxy[x + 1][y])
                        this->output[x][y] = 0;
                }
                else if (theta > 22.5 && theta <= 67.5) {
                    if (center < gxy[x - 1][y - 1] || center < gxy[x + 1][y + 1])
                        this->output[x][y] = 0;
                }
                else if (theta > 67.5 && theta <= 112.5) {
                    if (center < gxy[x][y - 1] || center < gxy[x][y + 1])
                        this->output[x][y];
                }
                else if (theta > 112.5 && theta <= 157.5) {
                    if (center < gxy[x + 1][y - 1] || center < gxy[x - 1][y + 1])
                        this->output[x][y] = 0;
                }
            }
        }
    }

	return;
}

double** nonMaxSup::copyGXYtoOutput(double** gxy, int height, int width) {
    double** newMatrix;

    newMatrix = (double**)malloc(sizeof(double*) * height);
    if (newMatrix == NULL) {
        std::cout << "Error allocating memory" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < height; i++) {
        newMatrix[i] = (double*)malloc(sizeof(double) * width);
        if (newMatrix[i] == NULL) {
            std::cout << "Error allocating memory" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            newMatrix[i][j] = gxy[i][j];
        }
    }

    return newMatrix;
}

void nonMaxSup::deallocateMatrix(int rows) {
    for (int i = 0; i < rows; i++) {
        free(this->output[i]);
    }
    free(this->output);

    return;
}