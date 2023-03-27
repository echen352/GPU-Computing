#include <iostream>
#include <cmath>
#include <stdlib.h>

class nonMaxSup
{
public:
	nonMaxSup();
	double* output;
	void nonMaxSuppression(double* gxy, double* iangle, int gxyHeight, int gxyWidth);
	void deallocateVector();
};

nonMaxSup::nonMaxSup() {
	output = NULL;
}

void nonMaxSup::nonMaxSuppression(double* gxy, double* iangle, int gxyHeight, int gxyWidth) {
	double theta;
	double center;

	this->output = (double*)malloc(sizeof(double) * gxyHeight * gxyWidth);
	for (int i = 0; i < gxyHeight * gxyWidth; i++)
		this->output[i] = gxy[i];

	for (int x = 0; x < gxyHeight; x++) {
		for (int y = 0; y < gxyWidth; y++) {
			theta = iangle[x * gxyWidth + y];
			if (theta < 0)
				theta += M_PI;
			theta = theta * (180 / M_PI);
			if (x - 1 > -1 && x + 1 < gxyHeight && y - 1 > -1 && y + 1 < gxyWidth) {
				center = gxy[x * gxyWidth + y];
				if (theta <= 22.5 || theta > 157.5) {
					if (center < gxy[(x - 1) * gxyWidth + y] || center < gxy[(x + 1) * gxyWidth + y])
						this->output[x * gxyWidth + y] = 0;
				}
				else if (theta > 22.5 && theta <= 67.5) {
					if (center < gxy[(x - 1) * gxyWidth + (y - 1)] || center < gxy[(x + 1) * gxyWidth + (y + 1)])
						this->output[x * gxyWidth + y] = 0;
				}
				else if (theta > 67.5 && theta <= 112.5) {
					if (center < gxy[x * gxyWidth + (y - 1)] || center < gxy[x * gxyWidth + (y + 1)])
						this->output[x * gxyWidth + y] = 0;
				}
				else if (theta > 112.5 && theta <= 157.5) {
					if (center < gxy[(x + 1) * gxyWidth + (y - 1)] || center < gxy[(x - 1) * gxyWidth + (y + 1)])
						this->output[x * gxyWidth + y] = 0;
				}
			}
		}
	}

	return;
}

void nonMaxSup::deallocateVector() {
    free(this->output);
    return;
}
