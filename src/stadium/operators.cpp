#include "spectral.hpp"
#include "stadium/geometry.hpp"
#include "stadium/operators.hpp"

DerivativeMatrices build_derivatives(const StadiumGeometry& g) {
	const int M = g.M;
	const int N = g.N;
	const double L = g.L;
	const arma::mat inv_r = arma::diagmat(1.0 / g.r);
	const arma::mat inv_r2 = arma::diagmat(1.0 / (g.r % g.r));

    DerivativeMatrices dv;

	arma::mat I_2M = arma::eye(2*M+1, 2*M+1);
	arma::mat I_M = arma::eye(M+1, M+1);
	arma::mat I_N = arma::eye(N+1, N+1);

	// First derivatives
    dv.D_r = chebyshev_diff_matrix(M, 0.0, 0.5 * L);
    dv.D_theta_left = chebyshev_diff_matrix(N, 0, pi);
    dv.D_theta_right = chebyshev_diff_matrix(N, pi, 2*pi);
    dv.D_x = chebyshev_diff_matrix(2*M, -0.5*L, 0.5*L);
    dv.D_y_upper = chebyshev_diff_matrix(M, 0.0, 0.5*L);
    dv.D_y_lower = chebyshev_diff_matrix(M, -0.5*L, 0.0);

	// Second derivatives
    dv.D2_r = dv.D_r * dv.D_r;
    dv.D2_theta_left = dv.D_theta_left * dv.D_theta_left;
    dv.D2_theta_right = dv.D_theta_right * dv.D_theta_right;
    dv.D2_x = dv.D_x * dv.D_x;
    dv.D2_y_upper = dv.D_y_upper * dv.D_y_upper;
    dv.D2_y_lower = dv.D_y_lower * dv.D_y_lower;

	// Laplacians on each domain
	dv.Lap_left  = 
		arma::kron(dv.D2_r, I_N) + 
		arma::kron(inv_r * dv.D_r, I_N) + 
		arma::kron(inv_r2 * I_M, dv.D2_theta_left);
	dv.Lap_right  = 
		arma::kron(dv.D2_r, I_N) + 
		arma::kron(inv_r * dv.D_r, I_N) + 
		arma::kron(inv_r2 * I_M, dv.D2_theta_right);
	dv.Lap_upper = arma::kron(dv.D2_x, I_M) + arma::kron(I_2M, dv.D2_y_upper);
	dv.Lap_lower = arma::kron(dv.D2_x, I_M) + arma::kron(I_2M, dv.D2_y_lower);

    return dv;
}

Hamiltonian build_hamiltonian(const StadiumGeometry& g) {
	const int N_r = g.M+1;
	const int N_theta = g.N+1;
	const int N_x = 2*g.M + 1;
	const int N_y = g.M + 1;
	DerivativeMatrices dv = build_derivatives(g);

	Hamiltonian h;
	h.h_left = dv.Lap_left;
	h.h_right = dv.Lap_right;
	h.h_upper = dv.Lap_upper;
	h.h_lower = dv.Lap_lower;

	//// row replacement
	// f(r = L, theta) = 0, r_grid[0] = L
	int k;
	arma::vec new_row;
	const int i_rmax = 0;
	for (int j_theta = 0; j_theta < N_theta; j_theta++) { 
		k = kron_index(N_r, N_theta, i_rmax, j_theta);
		new_row = arma::zeros<arma::vec>(N_r * N_theta);
		new_row(k) = 1;
		h.h_left.row(k) = new_row.t();
		h.h_right.row(k) = new_row.t();
		// TODO: row replace RHS
	}

	// f(x, y = +L/2) = 0, y_upper_grid[0] = +L/2
	// f(x, y = -L/2) = 0, y_lower_grid[-1] = -L/2
	const int j_ymax = 0;
	const int j_ymin = N_y-1;
	for (int i_x = 0; i_x < N_x; i_x++) {
		// upper
		k = kron_index(N_x, N_y, i_x, j_ymax);
		new_row = arma::zeros<arma::vec>(N_x * N_y);
		new_row(k) = 1;
		h.h_upper.row(k) = new_row.t();
		
		// lower
		k = kron_index(N_x, N_y, i_x, j_ymin);
		new_row = arma::zeros<arma::vec>(N_x * N_y);
		new_row(k) = 1;
		h.h_lower.row(k) = new_row.t();

		// TODO: row replace RHS
	}

	return h;
}
