#include "spectral.hpp"
#include <cmath>

arma::vec gauss_lobatto(int n, double ymin, double ymax) {
	arma::vec j = arma::regspace<arma::vec>(0, n);
	arma::vec x = arma::cos(pi * j / n);
	arma::vec y = ymin + (ymax - ymin) * (x + 1.0) / 2;
	return y;
}

arma::mat chebyshev_diff_matrix(int n, double ymin, double ymax) {
	arma::vec p = arma::ones<arma::vec>(n+1);
	p(0) = p(n) = 2.;
	
	arma::vec x = gauss_lobatto(n, -1, 1);
	arma::mat D = arma::zeros<arma::mat>(n+1, n+1);

	for (int i = 0; i <= n; i++) {
		for (int j = 0; j <= n; j++) {
			if (i == 0 and j == 0) {
				D(i, j) = (1. + 2. * n * n) / 6.;
			} 
			else if (i == n and j == n) {
				D(i, j) = -(1. + 2. * n * n) / 6.;
			} 
			else if (i == j) {
				D(i, j) = -x(i) / (2. * (1 - x(i) * x(i)));
			} 
			else {
				double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0; // (-1)^(i+j)
				D(i, j) = sign * p(i) / (p(j) * (x(i) - x(j)));
			}
		}
	}
	double scale = 2.0 / (ymax - ymin);
	D = scale * D;
	return D;
}

arma::mat fourier_diff_matrix(int n, double ymin, double ymax) {
	arma::vec x = 2. * pi * arma::linspace(0., 1., n) / n;
	arma::mat D = arma::zeros<arma::mat>(n, n);

	for (int i = 0; i <= n; i++) {
		for (int j = 0; j <= n; j++) {
			if (i == j) {
				D(i, j) = 0.;
			} else {
				double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0; // (-1)^(i+j)
				D(i, j) = sign * 0.5 / std::tan(pi * (x(i) - x(j)) / 2);
			}
		}
	}
	double scale = 2. * pi / (ymax - ymin);
	D = scale * D;
	return D;
}

int kron_index(int N_x, int N_y, int x_i, const int y_j) {
	/* Returns the tensor product index of (x_i, y_j) in a space defined by 
	 * X \otimes Y from the x index x_i and the y index y_j
	 * X space has N_x points, Y space has N_y points */
	if (x_i < 0 or y_j < 0 or x_i >= N_x or y_j >= N_y)
		return -1;
	else
		return x_i * N_y + y_j;
}

