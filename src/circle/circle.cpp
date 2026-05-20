#include <iostream>
#include <cmath>

#include "spectral.hpp"
#include "results.hpp"
#include "circle/circle.hpp"

Results solve_circle(int M, int N) {
	using namespace arma;

	const double rho_min = 0.;
	const double rho_max = 1.;
	const double theta_min = 0.;
	const double theta_max =  2.*pi;

	////////////////////
	// Define geometry
	////////////////////
	const vec rho = gauss_lobatto(M, rho_min, rho_max);
	const vec theta = 2.0 * pi * regspace(0., 1., N-1) / N;
	const unsigned n_rho = rho.n_rows; // M+1
	const unsigned n_theta = theta.n_rows; // N
	
	const vec rho_full = kron(rho, ones(n_theta));
	const vec theta_full = kron(ones(n_rho), theta);

	const uvec idx_rho_min = find(rho_full == rho_min); // n_theta degenerate origin points
	const uvec idx_rho_max = find(rho_full == rho_max);
	
	////////////////////
	// Define operators
	////////////////////
	const mat inv_rho = diagmat(1.0 / rho);
	const mat inv_rho2 = diagmat(1.0 / (rho % rho));

	const mat Irho = eye(n_rho, n_rho);
	const mat Itheta = eye(n_theta, n_theta);

	// first derivatives
	const mat Drho = chebyshev_diff_matrix(M, rho_min, rho_max);
	const mat Dtheta = fourier_diff_matrix(N, theta_min, theta_max);

	// second derivatives
	const mat D2rho = Drho * Drho;
	const mat D2theta = Dtheta * Dtheta;

	// Laplacian and Hamiltonian
	mat Lap =
		kron(D2rho + inv_rho * Drho, Itheta) +
		kron(inv_rho2 * Irho, D2theta);
	// regularize at the origin
	// del^2 u(0,0) = (u_xx + u_yy) (0,0)= u_rr(0, theta = {0, pi}) + u_rr(theta = {pi/2, 3pi/2})
	const unsigned int j_theta_0 = index_min(abs(theta));
	const unsigned int j_theta_pi2 = index_min(abs(theta - pi/2));
	const unsigned int j_theta_pi = index_min(abs(theta - pi));
	const unsigned int j_theta_3pi2 = index_min(abs(theta - 3.*pi/2));
	const uvec cardinal_idx = {j_theta_0, j_theta_pi2, j_theta_pi, j_theta_3pi2};

	for (int i = 0; i < 4; i++) {
		int j = cardinal_idx(i);
		if (std::abs(theta(j) - i*pi/2) > 1e-9) {
			std::cout << "WARNING: Your grid does not contain theta = " << i << "pi/2." << 
				"Results will probably not make sense.\n";
			std::cout << "  Rerun with N mod 4 = 0 to place grid points at theta = n*pi/2\n";
		}
	}

	const mat D2rho_full = kron(D2rho, Itheta);
	const rowvec u_xx = 0.5 * (
			D2rho_full.row(idx_rho_min(j_theta_0)) + 
			D2rho_full.row(idx_rho_min(j_theta_pi))
			); // u_xx = d^2/drho^2 | theta = {0,pi} averaged
	const rowvec u_yy = 0.5 * (
			D2rho_full.row(idx_rho_min(j_theta_pi2)) + 
			D2rho_full.row(idx_rho_min(j_theta_3pi2))
			); // u_yy = d^2/drho^2 | theta = {+/- pi/2} averaged
	for (auto k : idx_rho_min) {
		Lap.row(k) = u_xx + u_yy;
	}
	mat H = -Lap;
	std::cout << "H size: " << H.n_rows << " x " << H.n_cols << "\n";
	std::cout << "H has NaN: " << H.has_nan() << "\n";
	std::cout << "H has Inf: " << H.has_inf() << "\n";
	
	////////////////////
	// Generalized eigenvalue problem
	// Au = \lambda*Bu
	////////////////////
	mat A = H;
	mat B = eye(size(A));

	// Dirichlet BC: f(r = 1, theta) = 0
	A.rows(idx_rho_max).zeros();
	B.rows(idx_rho_max).zeros();
	for (auto k : idx_rho_max) {
		A(k, k) = 1.0;	
	}

	// build projection matrix averaging over all rows at the origin
	mat P = eye(size(A));
	vec avg = zeros<vec>(A.n_rows);
	for (auto k : idx_rho_min)
		avg(k) = 1.0 / std::sqrt((double)idx_rho_min.n_elem);
	P.col(idx_rho_min(0)) = avg;
	const uvec idx_redundant = idx_rho_min.tail(idx_rho_min.n_elem - 1);
	P.shed_cols(idx_redundant);
	const mat A_reduced = P.t() * A * P;
	const mat B_reduced = P.t() * B * P;
	auto [eigval, eigvec] = diagonalize_pair(A_reduced, B_reduced);

	////////////////////
	// Build results
	////////////////////
	vec x_full = rho_full % cos(theta_full);
	vec y_full = rho_full % sin(theta_full);
	x_full.shed_rows(idx_redundant);
	y_full.shed_rows(idx_redundant);
	A_reduced.save("A.csv", csv_ascii);
	B_reduced.save("B.csv", csv_ascii);
	Results r;
	r.geometry = "circle";
	r.x = x_full;
	r.y = y_full;
	r.eigval = eigval;
	r.eigvec = eigvec;
	r.excluded_points = zeros(r.x.n_rows);

	return r;
}
