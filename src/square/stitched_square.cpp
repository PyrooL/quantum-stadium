#include <iostream>
#include <cmath>
#include <armadillo>

#include "spectral.hpp"
#include "eigen.hpp"
#include "square/stitched_square.hpp"
#include "results.hpp"

struct Geometry {
	arma::vec x;
	arma::vec y_lower;
	arma::vec y_upper;
	int n_x;
	int n_y;
	arma::vec x_full;
	arma::vec y_full;
};

arma::mat stitched_square_hamiltonian(const Geometry g) {
	using namespace arma;

	// unpack struct for cleanliness
	const auto& n_x = g.n_x;
	const auto& n_y = g.n_y;
	const auto& x = g.x;
	const auto& y_lower = g.y_lower;
	const auto& y_upper = g.y_upper;

	const mat I_x = eye(n_x, n_x);
	const mat I_y = eye(n_y, n_y);

	// first derivatives
	const mat D_x = chebyshev_diff_matrix(n_x - 1, min(x), max(x));
	const mat D_y_lower = chebyshev_diff_matrix(n_y - 1, min(y_lower), max(y_lower));
	const mat D_y_upper = chebyshev_diff_matrix(n_y - 1, min(y_upper), max(y_upper));

	// second derivatives
	const mat D2_x = D_x * D_x;
	const mat D2_y_lower = D_y_lower * D_y_lower;
	const mat D2_y_upper = D_y_upper * D_y_upper;
	
	// Laplacian
	const mat Lap_lower = kron(D2_x, I_y) + kron(I_x, D2_y_lower);
	const mat Lap_upper = kron(D2_x, I_y) + kron(I_x, D2_y_upper);
	
	mat H = zeros(2*n_x*n_y, 2*n_x*n_y);
	H.submat(0, 0, n_x*n_y - 1, n_x*n_y - 1) = Lap_lower;
	H.submat(n_x*n_y, n_x*n_y, 2*n_x*n_y-1, 2*n_x*n_y - 1) = Lap_upper;
	std::cout << "H is hermitian: " << H.is_hermitian() << std::endl; // 0, why?

	// stitch along boundary
	const double lambda = 10;
	arma::mat V = arma::zeros(arma::size(H));

	const int j_y0_lower = index_max(y_lower);
	const int j_y0_upper = index_min(y_upper);
	for (int i_x = 0; i_x < n_x; i_x++) {
		int k_lower = kron_index(n_x, n_y, i_x, j_y0_lower);
		int k_upper = n_x*n_y + kron_index(n_x, n_y, i_x, j_y0_upper);
		V(k_lower, k_lower)++;
		V(k_upper, k_upper)++;
		V(k_lower, k_upper)--;
		V(k_upper, k_lower)--;
	}
	const mat H_full = H + lambda * V;

	return H_full;
}

arma::vec stitched_square_bc(const Geometry g) {
	using namespace arma;

	vec bc_points = zeros(2*g.n_x*g.n_y);
	bc_points(find(g.x_full == min(g.x_full))).ones();
	bc_points(find(g.x_full == max(g.x_full))).ones();
	bc_points(find(g.y_full == min(g.y_full))).ones();
	bc_points(find(g.y_full == max(g.y_full))).ones();
	return bc_points;
}

Results solve_stitched_square(int M, int N) {
	using namespace arma;

	Results r;
	r.geometry = "square";

	const float L_x = 1.0;
	const float L_y = 1.0;

	Geometry g;
	g.x = gauss_lobatto(M, -L_x/2, L_x/2);
	g.y_lower = gauss_lobatto(N, -L_y/2, 0.);
	g.y_upper = gauss_lobatto(N, 0., L_y/2);
	g.n_x = g.x.n_rows;
	g.n_y = g.y_lower.n_rows;
	g.x_full = join_cols(kron(g.x, ones(g.n_y)), kron(g.x, ones(g.n_y)));
	g.y_full = join_cols(kron(ones(g.n_x), g.y_lower), kron(ones(g.n_x), g.y_upper));


	mat H = stitched_square_hamiltonian(g);
	r.excluded_points = stitched_square_bc(g);

	const uvec excluded_indices = find(r.excluded_points);
	H.shed_rows(excluded_indices);
	H.shed_cols(excluded_indices);

	std::cout << "Built " << r.geometry << " hamiltonian\n";
	std::cout << "H size: " << H.n_rows << " x " << H.n_cols << std::endl;
	std::cout << "H has NaN: " << H.has_nan() << std::endl;
	std::cout << "H has Inf: " << H.has_inf() << std::endl;

	auto [eigval, eigvec] = diagonalize(H);
	r.x = g.x_full;
	r.y = g.y_full;
	r.eigval = eigval;
	r.eigvec = eigvec;

	return r;
}
