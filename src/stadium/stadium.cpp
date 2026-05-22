#include <iostream>
#include <cmath>

#include "spectral.hpp"
#include "stadium/stadium.hpp"
#include "results.hpp"

Results solve_stadium(int M, int N) {
	using namespace arma;
	
	////////////////////
	// Define geometry
	////////////////////
	const double L = 1.;
	const vec rho = gauss_lobatto(M, 0., L);
	const vec theta1 = gauss_lobatto(N, 0, pi);
	const vec theta4 = gauss_lobatto(N, pi, 2*pi);
	const vec x = gauss_lobatto(2*M, -L, L);
	const vec y2 = gauss_lobatto(M, 0., L);
	const vec y3 = gauss_lobatto(M, -L, 0.);

	const unsigned n_rho = rho.n_elem; // M+1
	const unsigned n_theta = theta1.n_elem; // N+1
	const unsigned n_x = x.n_elem; // 2M+1
	const unsigned n_y = y2.n_elem; // M+1
	
	// offset indices
	const unsigned n1 = n_rho*n_theta;
	const unsigned n2 = n1 + n_x*n_y;
	const unsigned n3 = n2 + n_x*n_y;
	const unsigned n4 = n3 + n_x*n_y;
	// const unsigned n_tot  = 2*(n_rho*n_theta + n_x*n_y);

	// subdomain kronecker coords
	const vec rho_full = kron(rho, ones(n_theta));
	const vec theta1_full = kron(ones(n_rho), theta1);
	const vec theta4_full = kron(ones(n_rho), theta4);
	const vec x_full = kron(x, ones(n_y));
	const vec y2_full = kron(ones(n_x), y2);
	const vec y3_full = kron(ones(n_x), y3);

	// useful indices
	const vec idx_pole = find(rho_full == L);

	////////////////////
	// Define operators
	////////////////////
	const mat Irho = eye(n_rho, n_rho);
	const mat Itheta = eye(n_theta, n_theta);
	const mat Ix = eye(n_x, n_x);
	const mat Iy = eye(n_y, n_y);

	const mat Drho = chebyshev_diff_matrix(M, 0., L);
	const mat Dtheta = chebyshev_diff_matrix(N, 0., pi);
	const mat Dx = chebyshev_diff_matrix(2*M, -L, L);
	const mat Dy = chebyshev_diff_matrix(M, 0., L);

	const mat D2rho = Drho * Drho;
	const mat D2theta = Dtheta * Dtheta;
	const mat D2x = Dx * Dx;
	const mat D2y = Dy * Dy;

	const mat inv_rho = diagmat(1.0 / rho);
	const mat inv_rho2 = diagmat(1.0 / (rho % rho));
	mat Lap1 = kron(D2rho + inv_rho*Drho, Itheta) + kron(inv_rho2 * Irho, D2theta); // also = Lap4
	mat Lap2 = kron(D2x, Iy) + kron(Ix, D2y); // also = Lap3

	// regularize polar laplacian at the origin
	const mat Drho_full = kron(Drho, Itheta);
	const mat D2theta = kron(Irho, D2theta);
	for (auto k: idx_pole) {
		Lap1.row(k) = D2theta_full.row(k) - 2.*pi/n_theta*Drho_full.row(k);
	}

	////////////////////
	// Build results
	////////////////////
	// Combine subdomain 
	const vec X = join_cols(-rho_full % sin(theta1_full) - 1, x_full, x_full, -rho_full % sin(theta4_full) + 1);
	const vec Y = join_cols(rho_full % cos(theta1_full), y2_full, y3_full, rho_full % cos(theta4_full));

	Results r;
	r.geometry = "stadium";
	r.x = X;
	r.y = Y;
	r.eigval = zeros<cx_vec>(r.x.n_rows); 
	r.eigvec = zeros<cx_mat>(r.x.n_rows, r.x.n_rows);
	r.excluded_points = zeros(r.x.n_rows);
	return r;

}


