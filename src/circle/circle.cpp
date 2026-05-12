#include <iostream>
#include <cmath>
#include <armadillo>

#include "spectral.hpp"
#include "eigen.hpp"
#include "results.hpp"
#include "circle/circle.hpp"

Results solve_circle(int M, int N) {
	Results r;
	r.geometry = "circle";

	const double lambda = 1e6; // strength of origin-matching potential
	const double rho_min = 0.;
	const double rho_max = 1.;
	const double theta_min = 0.;
	const double theta_max =  2.*pi;

	const arma::vec rho = gauss_lobatto(M, rho_min, rho_max);
	const arma::vec theta = 2.0 * pi * arma::regspace(0., 1., N-1) / N;

	const int n_rho = rho.n_rows; // M+1
	const int n_theta = theta.n_rows; // N
	
	// Build 2D x, y grids
	arma::mat R = arma::kron(rho, arma::ones(n_theta));
	arma::mat Theta = arma::kron(arma::ones(n_rho), theta);
	r.x = R % arma::cos(Theta);
	r.y = R % arma::sin(Theta);

	const arma::mat inv_rho = arma::diagmat(1.0 / rho);
	const arma::mat inv_rho2 = arma::diagmat(1.0 / (rho % rho));

	const arma::mat I_rho = arma::eye(n_rho, n_rho);
	const arma::mat I_theta = arma::eye(n_theta, n_theta);

	// first derivatives
	const arma::mat D_rho = chebyshev_diff_matrix(M, rho_min, rho_max);
	const arma::mat D_theta = fourier_diff_matrix(N, theta_min, theta_max);

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
	const int i_rho_min = arma::index_min(rho);
	const int i_rho_max = arma::index_max(rho);
	const int j_theta_0 = arma::index_min(arma::abs(theta));
	const int j_theta_pi2 = arma::index_min(arma::abs(theta - pi/2));
	const int j_theta_pi = arma::index_min(arma::abs(theta - pi));
	const int j_theta_3pi2 = arma::index_min(arma::abs(theta - 3.*pi/2));

	if (std::abs(theta(j_theta_0)) > 1e-9) {
		std::cout << "WARNING: Your grid does not contain theta = 0. Results will probably not make sense. Rerun with N % 4 = 0 to place a grid point at theta = n*pi/2\n";
	}
	if (std::abs(theta(j_theta_pi2) - pi/2) > 1e-9) {
		std::cout << "WARNING: Your grid does not contain theta = pi/2. Results will probably not make sense. Rerun with N % 4 = 0 to place a grid point at theta = pi/2\n";
	}
	if (std::abs(theta(j_theta_pi) - pi) > 1e-9) {
		std::cout << "WARNING: Your grid does not contain theta = pi. Results will probably not make sense. Rerun with N % 4 = 0 to place a grid point at theta = pi\n";
	}
	if (std::abs(theta(j_theta_3pi2) - 3.*pi/2) > 1e-9) {
		std::cout << "WARNING: Your grid does not contain theta = 3pi/2. Results will probably not make sense. Rerun with N % 4 = 0 to place a grid point at theta = 3pi/2\n";
	}

	arma::vec origin_idx = arma::zeros(n_theta);
	for (int j_origin = 0; j_origin < n_theta; j_origin++) {
		origin_idx(j_origin) = kron_index(n_rho, n_theta, i_rho_min, j_origin);
	}

	const arma::mat D2_rho_full_space = arma::kron(D2_rho, I_theta);
	const arma::rowvec u_xx = 0.5 * (
			D2_rho_full_space.row(origin_idx(j_theta_0)) + 
			D2_rho_full_space.row(origin_idx(j_theta_pi))
			); // u_xx = d^2/drho^2 | theta = {0,pi} averaged
	const arma::rowvec u_yy = 0.5 * (
			D2_rho_full_space.row(origin_idx(j_theta_pi2)) + 
			D2_rho_full_space.row(origin_idx(j_theta_3pi2))
			); // u_yy = d^2/drho^2 | theta = {+/- pi/2} averaged
	const arma::rowvec origin_laplacian = u_xx + u_yy;

	for (int i = 0; i < n_theta; i++) {
		H.row(origin_idx(i)) = origin_laplacian;
	}
	
	// add a potential to enforce equality of all origin indices
	arma::mat V = arma::zeros(arma::size(H));
	for (int i = 0; i < n_theta; i++) {
		int a = origin_idx(i);
		for (int j = i + 1; j < n_theta; j++) {
			int b = origin_idx(j);
			V(a, a)++;
			V(b, b)++;
			V(a, b)--;
			V(b, a)--;
		}
	}
	H += lambda * V;

	// row replacement at boundaries
	r.excluded_points = arma::zeros(H.n_rows);

	// Outer boundary: f(r = 1, theta) = 0
	for (int j_theta = 0; j_theta < n_theta; j_theta++) {
		int k = kron_index(n_rho, n_theta, i_rho_max, j_theta);
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
