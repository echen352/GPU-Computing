#include <cmath>

class Gauss
{
private:
	int w;
public:
	Gauss();
	double* g;
	double* g_deriv;
	double* gaussian(double sigma);
	double* gaussianDeriv(double sigma);
	int getGaussianLength();
	void deallocateMatrix();
};

Gauss::Gauss() {
	g = NULL;
	g_deriv = NULL;
	w = 0;
}

double* Gauss::gaussian(double sigma) {
	int a = round(2.5 * sigma - 0.5);
	this->w = 2 * a + 1;
	double sum = 0;

	this->g = new double[this->w];

	for (int i = 0; i < this->w; i++) {
		this->g[i] = exp((-1 * (i - a) * (i - a)) / (2 * sigma * sigma));
		sum += this->g[i];
	}

	for (int i = 0; i < this->w; i++) {
		this->g[i] /= sum;
	}

	return this->g;
}

double* Gauss::gaussianDeriv(double sigma) {
	int a = round(2.5 * sigma - 0.5);
	this->w = 2 * a + 1;
	double sum = 0;

	this->g_deriv = new double[this->w];

	for (int i = 0; i < this->w; i++) {
		this->g_deriv[i] = -1 * (i - a) * exp((-1 * (i - a) * (i - a)) / (2 * sigma * sigma));
		sum -= i * this->g_deriv[i];
	}

	for (int i = 0; i < this->w; i++) {
		this->g_deriv[i] /= sum;
	}

	return this->g_deriv;
}

int Gauss::getGaussianLength() {
	return this->w;
}

void Gauss::deallocateMatrix() {
	delete(this->g);
	delete(this->g_deriv);
}