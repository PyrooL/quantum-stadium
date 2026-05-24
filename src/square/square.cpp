#include <iostream>
#include <cmath>
#include <armadillo>

#include "spectral.hpp"
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

Results solve_stitched_square(int M) {
	using namespace arma;

	const float L_x = 1.0;
	const float L_y = 1.0;

	////////////////////
	// Define geometry
	////////////////////
	const vec x = gauss_lobatto(2*M, -L_x/2, L_x/2);
	const vec y_lower = gauss_lobatto(M, -L_y/2, 0.);
	const vec y_upper = gauss_lobatto(M, 0., L_y/2);
	const unsigned int n_x = x.n_rows;
	const unsigned int n_y = y_lower.n_rows;

	const vec x_full = join_cols(kron(x, ones(n_y)), kron(x, ones(n_y)));
	const vec y_full = join_cols(kron(ones(n_x), y_upper), kron(ones(n_x), y_lower));

	const uvec i_x_min = find(x_full == min(x_full));
	const uvec i_x_max = find(x_full == max(x_full));
	const uvec j_y_min = find(y_full == min(y_full));
	const uvec j_y_max = find(y_full == max(y_full));

	const uvec idx_interface = find(y_full == 0.);
	const uvec idx_interface_upper = idx_interface.head(n_x);
	const uvec idx_interface_lower = idx_interface.tail(n_x);

	////////////////////
	// Define operators
	////////////////////
	const mat Ix = eye(n_x, n_x);
	const mat Iy = eye(n_y, n_y);

	// first derivatives
	const mat Dx = chebyshev_diff_matrix(n_x - 1, min(x), max(x));
	const mat Dy = chebyshev_diff_matrix(n_y - 1, min(y_lower), max(y_lower));

	// second derivatives
	const mat D2x = Dx * Dx;
	const mat D2y = Dy * Dy;

	// Laplacian & Hamiltonian
	const mat Lap = kron(D2x, Iy) + kron(Ix, D2y);
	const mat H = -block_diag({Lap, Lap});
	std::cout << "H size: " << H.n_rows << " x " << H.n_cols << std::endl;
	std::cout << "H has NaN: " << H.has_nan() << std::endl;
	std::cout << "H has Inf: " << H.has_inf() << std::endl;
	
	////////////////////
	// Generalized eigenvalue problem
	// Au = \lambda*Bu
	////////////////////
	mat A = H;
	mat B = eye(size(A));

	// Dirichlet BCs
	const uvec boundary_points_idx  = join_cols(i_x_min, i_x_max, j_y_min, j_y_max);
	dirichlet_bc(A, B, boundary_points_idx);

	// Interface conditions
	const uvec idx_interface_upper_inner = idx_interface_upper.subvec(1, n_x - 2);
	const uvec idx_interface_lower_inner = idx_interface_lower.subvec(1, n_x - 2);
	const mat Dy_full = block_diag({kron(Ix, Dy), kron(Ix, Dy)});
	stitch_interface(A, B, idx_interface_upper_inner, idx_interface_lower_inner, Dy_full);

	////////////////////
	// Build results
	////////////////////
	auto [eigval, eigvec] = diagonalize_pair(A, B);
	Results r;
	r.geometry = "square";
	r.x = x_full;
	r.y = y_full;
	r.eigval = eigval;
	r.eigvec = eigvec;
	r.excluded_points = zeros(r.x.n_rows);

	return r;
}

