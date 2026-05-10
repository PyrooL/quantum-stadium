#include <iostream>
#include <cmath>
#include <armadillo>

#include "spectral.hpp"
#include "eigen.hpp"
#include "results.hpp"
#include "semicircle/semicircle.hpp"

Results solve_semicircle(int M, int N) {
	Results r;
	r.geometry = "semicircle";

	// const float lambda = 100.; // strength of origin-matching potential
	const float rho_min = 0.;
	const float rho_max = 1.;
	const float theta_min = -pi/2;
	const float theta_max =  pi/2;

	const arma::vec rho = gauss_lobatto(M, rho_min, rho_max);
	const arma::vec theta = gauss_lobatto(N, theta_min, theta_max);
	const int n_rho = rho.n_rows; // M+1
	const int n_theta = theta.n_rows; // N+1
	
	// Build 2D x, y grids
	arma::mat R     = arma::kron(rho, arma::ones(n_theta));
	arma::mat Theta = arma::kron(arma::ones(n_rho), theta);
	r.x = R % arma::cos(Theta);
	r.y = R % arma::sin(Theta);

	const arma::mat inv_rho = arma::diagmat(1.0 / rho);
	const arma::mat inv_rho2 = arma::diagmat(1.0 / (rho % rho));

	const arma::mat I_rho = arma::eye(n_rho, n_rho);
	const arma::mat I_theta = arma::eye(n_theta, n_theta);

	// first derivatives
	const arma::mat D_rho = chebyshev_diff_matrix(M, rho_min, rho_max);
	const arma::mat D_theta = chebyshev_diff_matrix(N, theta_min, theta_max);

	// second derivatives
	const arma::mat D2_rho = D_rho * D_rho;
	const arma::mat D2_theta = D_theta * D_theta;

	// Laplacian
	const arma::mat Lap =
		arma::kron(D2_rho, I_theta) + 
		arma::kron(inv_rho * D_rho, I_theta) +
		arma::kron(inv_rho2 * I_rho, D2_theta);

	arma::mat H = -Lap;
	
	// regularize at the origin
	// Δu(0,0) = (u_xx + u_yy) (0,0)= u_rr(0, theta = {0, pi}) + u_rr(theta = {pi/2, 3pi/2})
	// Here on the semicircle we only have -pi/2 < theta < pi/2
	const int i_rho_min = arma::index_min(rho);
	const int i_rho_max = arma::index_max(rho);
	const int j_theta_min = arma::index_min(theta);
	const int j_theta_max = arma::index_max(theta);
	const int j_theta_0 = arma::index_min(arma::abs(theta));

	if (!(std::abs(theta(j_theta_0)) < 1e-9)) {
		std::cout << "WARNING: Your grid does not contain theta = 0. Results will probably not make sense. Rerun with an even N to place a grid point at theta = 0\n";
	}

	arma::vec origin_idx = arma::zeros(n_theta);
	for (int j_origin = 0; j_origin < n_theta; j_origin++) {
		origin_idx(j_origin) = kron_index(n_rho, n_theta, i_rho_min, j_origin);
	}

	const arma::mat D2_rho_full_space = arma::kron(D2_rho, I_theta);
	const arma::rowvec u_xx = D2_rho_full_space.row(origin_idx(j_theta_0)); // u_xx = d^2/drho^2 | theta = 0
	const arma::rowvec u_yy = 0.5 * (
			D2_rho_full_space.row(origin_idx(j_theta_min)) + 
			D2_rho_full_space.row(origin_idx(j_theta_max))
			); // u_yy = d^2/drho^2 | theta = {+/- pi/2} averaged
	const arma::rowvec origin_laplacian = u_xx + u_yy;
	H.row(origin_idx(j_theta_0)) = origin_laplacian;
	
	/*
	// add a potential to enforce equality of all origin indices
	arma::mat V = arma::zeros(arma::size(H));
	for (int i = 0; i < origin_idx.n_rows; i++) {
		int a = origin_idx(i);
		for (int j = i + 1; j < origin_idx.n_rows, i++) {
			int b = origin_idx(j);
			V(a, a)++;
			V(b, b)++;
			V(a, b)--;
			V(b, a)--;
		}
	}
	H += lambda * V;
	*/

	// row replacement at boundaries
	r.excluded_points = arma::zeros(H.n_rows);
	int k;

	// Outer boundary: f(r = 1, theta) = 0
	// and origin f(r = 0, theta) = 0
	for (int j_theta = 0; j_theta < n_theta; j_theta++) {
		k = kron_index(n_rho, n_theta, i_rho_max, j_theta);
		r.excluded_points(k) = 1;

		k = kron_index(n_rho, n_theta, i_rho_min, j_theta);
		r.excluded_points(k) = 1;
	}
	
	// Inner boundaries: f(r, theta = +/- pi/2) = 0
	for (int i_rho = 0; i_rho < n_rho; i_rho++) {
		// theta = +pi/2
		k = kron_index(n_rho, n_theta, i_rho, j_theta_max);
		r.excluded_points(k) = 1;
	
		// theta = -pi/2
		k = kron_index(n_rho, n_theta, i_rho, j_theta_min);
		r.excluded_points(k) = 1;
	}
	
	arma::uvec excluded_indices = arma::find(r.excluded_points);
	H.shed_rows(excluded_indices);
	H.shed_cols(excluded_indices);

	std::cout << "Built " << r.geometry << " hamiltonian\n";
	std::cout << "H size: " << H.n_rows << " x " << H.n_cols << "\n";
	std::cout << "H has NaN: " << H.has_nan() << "\n";
	std::cout << "H has Inf: " << H.has_inf() << "\n";
	std::cout << "H is hermitian: " << H.is_hermitian() << std::endl;

	auto [eigval, eigvec] = diagonalize(H);
	r.eigval = eigval;
	r.eigvec = eigvec;

	return r;

}
