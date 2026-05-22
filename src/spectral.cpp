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
	D *= 2.0 / (ymax - ymin);
	return D;
}

arma::mat fourier_diff_matrix(int n, double ymin, double ymax) {
	arma::mat D = arma::zeros<arma::mat>(n, n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) continue;
			double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0; // (-1)^(i+j)
			double theta = pi * (i - j) / n;
			D(i, j) = sign * 0.5 / std::tan(theta);
			}
		}
	D *= 2. * pi / (ymax - ymin);
	return D;
}

// Linalg helpers 
int kron_index(int N_x, int N_y, int x_i, const int y_j) {
	/* Returns the tensor product index of (x_i, y_j) in a space defined by 
	 * X \otimes Y from the x index x_i and the y index y_j
	 * X space has N_x points, Y space has N_y points */
	if (x_i < 0 or y_j < 0 or x_i >= N_x or y_j >= N_y)
		return -1;
	else
		return x_i * N_y + y_j;
}


std::tuple<arma::cx_vec, arma::cx_mat> diagonalize(const arma::mat &M) {
	arma::cx_vec eigval;
	arma::cx_mat eigvec;
	bool diag_success = arma::eig_gen(eigval, eigvec, M);
	if (not diag_success) {
        throw std::runtime_error("Diagonalization failed");
	}
	return {eigval, eigvec};
}


std::tuple<arma::cx_vec, arma::cx_mat> diagonalize_pair(const arma::mat &A, const arma::mat &B) {
	arma::cx_vec eigval;
	arma::cx_mat eigvec;
	bool diag_success = arma::eig_pair(eigval, eigvec, A, B);
	if (not diag_success) {
        throw std::runtime_error("Diagonalization failed");
	}
	return {eigval, eigvec};
}


arma::mat block_diag(const arma::mat& A, const arma::mat& B) {
    const arma::uword rows = A.n_rows + B.n_rows;
    const arma::uword cols = A.n_cols + B.n_cols;
    arma::mat M(rows, cols, arma::fill::zeros);
    M.submat(0, 0, A.n_rows - 1, A.n_cols - 1) = A;
    M.submat(A.n_rows, A.n_cols, rows - 1, cols - 1) = B;
    return M;
}
