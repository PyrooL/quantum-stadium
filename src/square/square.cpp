#include <iostream>
#include <cmath>
#include <armadillo>

#include "spectral.hpp"
#include "eigen.hpp"
#include "square/square.hpp"
#include "results.hpp"

Results solve_square(int M, int N) {
	Results r;
	r.geometry = "square";

	const float L_x = 1.0;
	const float L_y = 1.0;

	const arma::vec x = gauss_lobatto(M, 0., L_x);
	const arma::vec y = gauss_lobatto(N, 0., L_y);
	const int n_x = x.n_rows;
	const int n_y = y.n_rows;
	r.x = arma::kron(x, arma::ones(n_y));
	r.y = arma::kron(arma::ones(n_x), y);

	const arma::mat I_x = arma::eye(n_x, n_x);
	const arma::mat I_y = arma::eye(n_y, n_y);

	// first derivatives
	const arma::mat D_x = chebyshev_diff_matrix(M, 0., L_x);
	const arma::mat D_y = chebyshev_diff_matrix(N, 0., L_y);
	// second derivatives
	const arma::mat D2_x = D_x * D_x;
	const arma::mat D2_y = D_y * D_y;
	// Laplacian
	const arma::mat Lap =
		arma::kron(D2_x, I_y) + 
		arma::kron(I_x, D2_y);

	arma::mat H = -Lap;

	// row replacement
	r.excluded_points = arma::zeros(H.n_rows);
	int k;

	const int i_x_max = arma::index_max(x);
	const int i_x_min = arma::index_min(x);
	const int j_y_max = arma::index_max(y);
	const int j_y_min = arma::index_min(y);

	for (int j_y = 0; j_y < n_y ; j_y++) {
		k = kron_index(n_x, n_y, i_x_max, j_y);
		r.excluded_points(k) = 1;

		k = kron_index(n_x, n_y, i_x_min, j_y);
		r.excluded_points(k) = 1;
	}
	
	for (int i_x = 0; i_x < n_x ; i_x++) {
		k = kron_index(n_x, n_y, i_x, j_y_max);
		r.excluded_points(k) = 1;

		k = kron_index(n_x, n_y, i_x, j_y_min);
		r.excluded_points(k) = 1;
	}

	const arma::uvec excluded_indices = arma::find(r.excluded_points);
	H.shed_rows(excluded_indices);
	H.shed_cols(excluded_indices);

	std::cout << "Built " << r.geometry << " hamiltonian\n";
	std::cout << "H size: " << H.n_rows << " x " << H.n_cols << std::endl;
	std::cout << "H has NaN: " << H.has_nan() << std::endl;
	std::cout << "H has Inf: " << H.has_inf() << std::endl;
	std::cout << "H is hermitian: " << H.is_hermitian() << std::endl;

	auto [eigval, eigvec] = diagonalize(H);
	r.eigval = eigval;
	r.eigvec = eigvec;

	return r;
}

