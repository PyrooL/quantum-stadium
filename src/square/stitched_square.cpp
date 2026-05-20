#include <iostream>
#include <cmath>

#include "spectral.hpp"
#include "results.hpp"
#include "square/stitched_square.hpp"


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
	const uvec boundary_points_idx  = join_cols(i_x_min, i_x_max, j_y_min, j_y_max);

	const uvec idx_interface = find(y_full == 0.);
	const uvec idx_interface_upper = idx_interface.head(n_x);
	const uvec idx_interface_lower = idx_interface.tail(n_x);
	const uvec idx_interface_upper_inner = idx_interface_upper.subvec(1, n_x - 2);
	const uvec idx_interface_lower_inner = idx_interface_lower.subvec(1, n_x - 2);

	////////////////////
	// Define operators
	////////////////////
	const mat Ix = eye(n_x, n_x);
	const mat Iy = eye(n_y, n_y);

	// first derivatives
	const mat Dx = chebyshev_diff_matrix(n_x - 1, min(x), max(x));
	const mat Dy_lower = chebyshev_diff_matrix(n_y - 1, min(y_lower), max(y_lower));
	const mat Dy_upper = chebyshev_diff_matrix(n_y - 1, min(y_upper), max(y_upper));
	const mat Dy = block_diag(kron(Ix, Dy_upper), kron(Ix, Dy_lower));

	// second derivatives
	const mat D2x = Dx * Dx;
	const mat D2y_lower = Dy_lower * Dy_lower;
	const mat D2y_upper = Dy_upper * Dy_upper;

	// Laplacian & Hamiltonian
	const mat Lap_lower = kron(D2x, Iy) + kron(Ix, D2y_lower);
	const mat Lap_upper = kron(D2x, Iy) + kron(Ix, D2y_upper);
	const mat H = -block_diag(Lap_upper, Lap_lower);
	std::cout << "H size: " << H.n_rows << " x " << H.n_cols << std::endl;
	std::cout << "H has NaN: " << H.has_nan() << std::endl;
	std::cout << "H has Inf: " << H.has_inf() << std::endl;
	
	////////////////////
	// Generalized eigenvalue problem
	// Au = \lambda*Bu
	////////////////////
	mat A = H;
	mat B = eye(size(A));

	// Interface conditions
	// Continuity of wave function
	A.rows(idx_interface_upper_inner).zeros();
	B.rows(idx_interface_upper_inner).zeros();
	for (unsigned int k = 0; k < n_x-2; k++) {
		A(idx_interface_upper_inner(k), idx_interface_upper_inner(k)) = 1.0;
		A(idx_interface_upper_inner(k), idx_interface_lower_inner(k)) = -1.0;
	}
	// Continuity of y derivative
	A.rows(idx_interface_lower_inner) = Dy.rows(idx_interface_upper_inner) - Dy.rows(idx_interface_lower_inner);
	B.rows(idx_interface_lower_inner).zeros();

	// Dirichlet BCs
	A.rows(boundary_points_idx).zeros();
	B.rows(boundary_points_idx).zeros();
	for (auto k : boundary_points_idx) {
		A(k, k) = 1.0;
	}

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

