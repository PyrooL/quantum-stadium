#include <iostream>
#include <cmath>
#include <armadillo>

#include "spectral.hpp"
#include "results.hpp"
#include "semicircle/semicircle.hpp"

Results solve_semicircle(int M, int N) {
	using namespace arma;

	const float rho_min = 0.;
	const float rho_max = 1.;
	const float theta_min = -pi/2;
	const float theta_max =  pi/2;

	////////////////////
	// Define geometry
	////////////////////
	const vec rho = gauss_lobatto(M, rho_min, rho_max);
	const vec theta = gauss_lobatto(N, theta_min, theta_max);
	const unsigned n_rho = rho.n_rows; // M+1
	const unsigned n_theta = theta.n_rows; // N+1
	
	const vec rho_full = kron(rho, ones(n_theta));
	const vec theta_full = kron(ones(n_rho), theta);

	const uvec idx_rho_min = find(rho_full == rho_min);
	const uvec idx_rho_max = find(rho_full == rho_max);
	const uvec idx_theta_min = find(theta_full == theta_min);
	const uvec idx_theta_max = find(theta_full == theta_max);
	
	////////////////////
	// Define operators
	////////////////////
	const mat inv_rho = diagmat(1.0 / rho);
	const mat inv_rho2 = diagmat(1.0 / (rho % rho));

	const mat Irho = eye(n_rho, n_rho);
	const mat Itheta = eye(n_theta, n_theta);

	// first derivatives
	const mat Drho = chebyshev_diff_matrix(M, rho_min, rho_max);
	const mat Dtheta = chebyshev_diff_matrix(N, theta_min, theta_max);

	// second derivatives
	const mat D2rho = Drho * Drho;
	const mat D2theta = Dtheta * Dtheta;

	// Laplacian
	mat Lap =
		kron(D2rho, Itheta) + 
		kron(inv_rho * Drho, Itheta) +
		kron(inv_rho2 * Irho, D2theta);

	// regularize at the origin
	// Δu(0,0) = (u_xx + u_yy) (0,0)= u_rr(0, theta = {0, pi}) + u_rr(theta = {pi/2, 3pi/2})
	// Here on the semicircle we only have -pi/2 < theta < pi/2
	const int j_theta_min = index_min(theta);
	const int j_theta_max = index_max(theta);
	const int j_theta_0 = n_theta / 2;

	if (!(std::abs(theta(j_theta_0)) < 1e-9)) {
		std::cout << "WARNING: Your grid does not contain theta = 0. Results will probably not make sense. Rerun with an even N to place a grid point at theta = 0\n";
	}

	const mat D2rho_full = kron(D2rho, Itheta);
	const rowvec u_xx = (
			D2rho_full.row(idx_rho_min(j_theta_0))
			); // u_xx = d^2/drho^2 | theta = 0
	const rowvec u_yy = 0.5 * (
			D2rho_full.row(idx_rho_min(j_theta_min)) + 
			D2rho_full.row(idx_rho_min(j_theta_max))
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

	A.rows(idx_rho_max).zeros();
	B.rows(idx_rho_max).zeros();
	for (auto k : idx_rho_max) {
		A(k, k) = 1.0;
	}

	A.rows(idx_theta_min).zeros();
	B.rows(idx_theta_min).zeros();
	for (auto k : idx_theta_min) {
		A(k, k) = 1.0;
	}

	A.rows(idx_theta_max).zeros();
	B.rows(idx_theta_max).zeros();
	for (auto k : idx_theta_max) {
		A(k, k) = 1.0;
	}
	
	// build projection matrix averaging over all rows at the origin
	mat P = eye(size(A));
	const uvec idx_redundant = idx_rho_min.tail(idx_rho_min.n_elem - 1);
	for (auto k : idx_rho_min)
		P(k, idx_rho_min(0)) = 1.;
	P.shed_cols(idx_redundant);
	const mat A_reduced = P.t() * A * P;
	const mat B_reduced = P.t() * B * P;
	auto [eigval, eigvec] = diagonalize_pair(A_reduced, B_reduced);

	////////////////////
	// Build results
	////////////////////
	vec x_full = rho_full % cos(theta_full);
	vec y_full = rho_full % sin(theta_full);
	A_reduced.save("A.csv", csv_ascii);
	B_reduced.save("B.csv", csv_ascii);

	Results r;
	r.geometry = "semicircle";
	r.x = P.t() * x_full;
	r.y = P.t() * y_full;
	r.eigval = eigval;
	r.eigvec = eigvec;
	r.excluded_points = zeros(r.x.n_rows);

	return r;

}
