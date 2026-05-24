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


arma::mat block_diag(const std::vector<arma::mat>& blocks)
{
    arma::uword total_rows = 0;
    arma::uword total_cols = 0;

    for (const auto& M : blocks) {
        total_rows += M.n_rows;
        total_cols += M.n_cols;
    }

    arma::mat out(total_rows, total_cols, arma::fill::zeros);

    arma::uword row_offset = 0;
    arma::uword col_offset = 0;

    for (const auto& M : blocks) {
        out.submat(
            row_offset,
            col_offset,
            row_offset + M.n_rows - 1,
            col_offset + M.n_cols - 1
        ) = M;

        row_offset += M.n_rows;
        col_offset += M.n_cols;
    }

    return out;
}

void stitch_interface(
		arma::mat& A, 
		arma::mat& B, 
		const arma::uvec& idx1, 
		const arma::uvec& idx2, 
		const arma::mat& D_interface) {
	const unsigned n = idx1.n_elem;
	A.rows(idx1).zeros();
	B.rows(idx1).zeros();
	for (unsigned i = 0; i < n; i++) {
		A(idx1(i), idx1(i)) = 1.0;
		A(idx1(i), idx2(i)) = -1.0;
	}
	A.rows(idx2) = D_interface.rows(idx1) - D_interface.rows(idx2);
	B.rows(idx2).zeros();
}

void dirichlet_bc(arma::mat& A, arma::mat& B, const arma::uvec& idx_boundary_points) {
	A.rows(idx_boundary_points).zeros();
	B.rows(idx_boundary_points).zeros();
	for (auto k: idx_boundary_points) {
		A(k,k) = 1;
	}
}
