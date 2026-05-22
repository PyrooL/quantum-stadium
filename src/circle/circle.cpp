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
	const double theta_max = 2.*pi;

	////////////////////
	// Define geometry
	////////////////////
	const vec rho = gauss_lobatto(M, rho_min, rho_max);
	// const vec theta = 2.*pi * regspace(0., 1., N-1) / N;
	const vec theta = gauss_lobatto(N, theta_min, theta_max);
	const unsigned n_rho = rho.n_rows; // M+1
	const unsigned n_theta = theta.n_rows; // N
	
	const vec rho_full = kron(rho, ones(n_theta));
	const vec theta_full = kron(ones(n_rho), theta);
	vec x_full = rho_full % cos(theta_full);
	vec y_full = rho_full % sin(theta_full);

	const uvec idx_rho_max = find(rho_full == rho_max);
	const uvec idx_rho_min = find(rho_full == rho_min); // origin points
	const uvec idx_theta_max = find(theta_full == theta_max);
	const uvec idx_theta_min = find(theta_full == theta_min);
	const uvec idx_theta_max_inner = idx_theta_max.subvec(1, n_rho - 2);
	const uvec idx_theta_min_inner = idx_theta_min.subvec(1, n_rho - 2);
	
	////////////////////
	// Define operators
	////////////////////
	const mat inv_rho = diagmat(1.0 / rho);
	const mat inv_rho2 = diagmat(1.0 / (rho % rho));

	const mat Irho = eye(n_rho, n_rho);
	const mat Itheta = eye(n_theta, n_theta);

	// first derivatives
	const mat Drho = chebyshev_diff_matrix(M, rho_min, rho_max);
	// const mat Dtheta = fourier_diff_matrix(N);
	const mat Dtheta = chebyshev_diff_matrix(N, theta_min, theta_max);

	// second derivatives
	const mat D2rho = Drho * Drho;
	const mat D2theta = Dtheta * Dtheta;

	// Laplacian
	mat Lap =
		kron(D2rho + inv_rho * Drho, Itheta) +
		kron(inv_rho2 * Irho, D2theta);
	// regularize at the origin
	// Huang & Sloan (1992) "Pole Condition for Singular Problems: The Pseudospectral Approximation". https://doi.org/10.1006/jcph.1993.1141
	const mat Drho_full = kron(Drho, Itheta);
	const mat D2rho_full = kron(D2rho, Itheta);
	const mat Dtheta_full = kron(Irho, Dtheta);
	const mat D2theta_full = kron(Irho, D2theta);
	for (auto k: idx_rho_min) {
		Lap.row(k) = D2theta_full.row(k) - 2.*pi/n_theta*Drho_full.row(k);
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

	B.rows(idx_rho_min).zeros();

	// Interface conditions
	// Continuity of wave function
	A.rows(idx_theta_max_inner).zeros();
	B.rows(idx_theta_max_inner).zeros();
	for (unsigned int k = 0; k < n_rho-2; k++) {
		A(idx_theta_max_inner(k), idx_theta_max_inner(k)) = 1.0;
		A(idx_theta_max_inner(k), idx_theta_min_inner(k)) = -1.0;
	}
	// Continuity of theta derivative
	A.rows(idx_theta_min_inner) = Dtheta_full.rows(idx_theta_max_inner) - Dtheta_full.rows(idx_theta_min_inner);
	B.rows(idx_theta_min_inner).zeros();

	auto [eigval, eigvec] = diagonalize_pair(A, B);
	A.save("A.csv", csv_ascii);
	B.save("B.csv", csv_ascii);

	////////////////////
	// Build results
	////////////////////
	Results r;
	r.geometry = "circle";
	r.x = x_full;
	r.y = y_full;
	r.eigval = eigval;
	r.eigvec = eigvec;
	r.excluded_points = zeros(r.x.n_rows);

	return r;
}

Results solve_circle_fourier(int M, int N) {
	using namespace arma;

	const double rho_min = 0.;
	const double rho_max = 1.;

	////////////////////
	// Define geometry
	////////////////////
	const vec rho = gauss_lobatto(M, rho_min, rho_max);
	const vec theta = 2.*pi * regspace(0., 1., N-1) / N;
	const unsigned n_rho = rho.n_rows; // M+1
	const unsigned n_theta = theta.n_rows; // N
	
	const vec rho_full = kron(rho, ones(n_theta));
	const vec theta_full = kron(ones(n_rho), theta);
	vec x_full = rho_full % cos(theta_full);
	vec y_full = rho_full % sin(theta_full);

	const uvec idx_rho_max = find(rho_full == rho_max);
	const uvec idx_rho_min = find(rho_full == rho_min); // origin points
	const uvec idx_redundant = idx_rho_min.tail(n_theta - 1);
	// const unsigned k0 = idx_rho_min(0);
	
	////////////////////
	// Define operators
	////////////////////
	const mat inv_rho = diagmat(1.0 / rho);
	const mat inv_rho2 = diagmat(1.0 / (rho % rho));

	const mat Irho = eye(n_rho, n_rho);
	const mat Itheta = eye(n_theta, n_theta);

	// first derivatives
	const mat Drho = chebyshev_diff_matrix(M, rho_min, rho_max);
	const mat Dtheta = fourier_diff_matrix(N);

	// second derivatives
	const mat D2rho = Drho * Drho;
	const mat D2theta = Dtheta * Dtheta;

	// Laplacian
	mat Lap =
		kron(D2rho + inv_rho * Drho, Itheta) +
		kron(inv_rho2 * Irho, D2theta);
	// regularize at the origin
	// Huang & Sloan (1992) "Pole Condition for Singular Problems: The Pseudospectral Approximation". https://doi.org/10.1006/jcph.1993.1141
	const mat Drho_full = kron(Drho, Itheta);
	const mat D2rho_full = kron(D2rho, Itheta);
	const mat Dtheta_full = kron(Irho, Dtheta);
	const mat D2theta_full = kron(Irho, D2theta);
	for (auto k: idx_rho_min) {
		Lap.row(k) = D2theta_full.row(k) - 2.*pi/n_theta*Drho_full.row(k);
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

	B.rows(idx_rho_min).zeros();

	auto [eigval, eigvec] = diagonalize_pair(A, B);
	A.save("A.csv", csv_ascii);
	B.save("B.csv", csv_ascii);

	////////////////////
	// Build results
	////////////////////
	Results r;
	r.geometry = "circle";
	r.x = x_full;
	r.y = y_full;
	r.eigval = eigval;
	r.eigvec = eigvec;
	r.excluded_points = zeros(r.x.n_rows);

	return r;
}
