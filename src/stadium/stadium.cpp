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

	// subdomain kronecker coords
	const vec rho_full = kron(rho, ones(n_theta));
	const vec theta1_full = kron(ones(n_rho), theta1);
	const vec theta4_full = kron(ones(n_rho), theta4);
	const vec x_full = kron(x, ones(n_y));
	const vec y2_full = kron(ones(n_x), y2);
	const vec y3_full = kron(ones(n_x), y3);
	
	// offset indices
	const unsigned n1 = 0;
	const unsigned n2 = n1 + rho_full.n_elem; // n_rho*n_theta;
	const unsigned n3 = n2 + x_full.n_elem; // n_x * n_y
	const unsigned n4 = n3 + x_full.n_elem; 

	// useful indices
	// Careful with the theta indices when indexing left and right semicircles!
	const uvec idx_rho_min = find(rho_full == 0.); // radial origin
	const uvec idx_rho_max = find(rho_full == L); // rho == L BC on both 1 and 4
	const uvec idx_theta_min = find(theta1_full == 0.); // 1->2 and 4->3 boundary
	const uvec idx_theta_max = find(theta1_full == pi); // 1->3 and 4->2 boundary

	const uvec idx_x_max = find(x_full == L);
	const uvec idx_x_min = find(x_full == -L);
	const uvec idx_y_max = find(y2_full == L); // y2 == L BC and y3 == 0 interface
	const uvec idx_y_min = find(y3_full == -L); // y3 == -L BC and y2 == 0 interface

	// Combine subdomain 
	const vec X = join_cols(
			-rho_full % sin(theta1_full) - 1, x_full, x_full, -rho_full % sin(theta4_full) + 1
			);
	const vec Y = join_cols(
			rho_full % cos(theta1_full), y2_full, y3_full, rho_full % cos(theta4_full)
			);
	const uvec idx_boundary_points = join_cols(
			n1 + idx_rho_max, n2 + idx_y_max, n3 + idx_y_min, n4 + idx_rho_max
			);

	////////////////////
	// Define operators
	////////////////////
	const mat Irho = eye(n_rho, n_rho);
	const mat Itheta = eye(n_theta, n_theta);
	const mat Ix = eye(n_x, n_x);
	const mat Iy = eye(n_y, n_y);
	const mat Ipolar = kron(Irho, Itheta);
	const mat Icart = kron(Ix, Iy);

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
	mat Lap_rad = kron(D2rho + inv_rho*Drho, Itheta) + kron(inv_rho2 * Irho, D2theta); // = Lap1 = Lap4
	mat Lap_cart = kron(D2x, Iy) + kron(Ix, D2y); // = Lap 2 = Lap3

	// regularize polar laplacian at the origin
	const mat Drho_full = kron(Drho, Itheta);
	const mat D2theta_full = kron(Irho, D2theta);
	for (auto k: idx_rho_min) {
		// pi or 2pi for half circle?
		Lap_rad.row(k) = D2theta_full.row(k) + 2.*pi/n_theta*Drho_full.row(k);
	}

	////////////////////
	// Generalized eigenvalue problem
	// Au = \lambda*Bu
	////////////////////
	mat A = -block_diag({Lap_rad, Lap_cart, Lap_cart, Lap_rad});
	mat B = eye(size(A));

	B.rows(idx_rho_min).zeros();
	B.rows(n3 + idx_rho_min).zeros();

	// Interface conditions
	// Region 2 <-> Region 3
	uvec idx_upper = n2 + idx_y_min;
	uvec idx_lower = n3 + idx_y_max;
	if (not approx_equal(X(idx_upper), X(idx_lower), "absdiff", 1e-9))
		join_rows(X(idx_upper), X(idx_lower)).print("X (2-3):");
	if (not approx_equal(Y(idx_upper), Y(idx_lower), "absdiff", 1e-9))
		join_rows(Y(idx_upper), Y(idx_lower)).print("Y (2-3):");
	mat D_interface = block_diag({Ipolar, kron(Ix, Dy), kron(Ix, Dy), Ipolar});
	stitch_interface(A, B, idx_upper, idx_lower, D_interface);

	// Region 1 <-> Region 2
	// Note the sign when matching x derivatives to theta derivatives
	uvec idx_left  = n1 + idx_theta_min.subvec(1, n_rho-2);
	uvec idx_right = n2 + idx_x_min.subvec(1, n_y-2);
	if (not approx_equal(X(idx_left), X(idx_right), "absdiff", 1e-9))
		join_rows(X(idx_left), X(idx_right)).print("Y (1-2):");
	if (not approx_equal(Y(idx_left), Y(idx_right), "absdiff", 1e-9))
		join_rows(Y(idx_left), Y(idx_right)).print("Y (1-2):");
	D_interface = block_diag({kron(inv_rho, -Dtheta), kron(Dx, Iy), Icart, Ipolar});
	stitch_interface(A, B, idx_left, idx_right, D_interface);

	// Region 1 <-> Region 3
	idx_left  = n1 + idx_theta_max.subvec(1, n_rho-2);
	idx_right = n3 + reverse(idx_x_min.subvec(1, n_y-2));
	if (not approx_equal(X(idx_left), X(idx_right), "absdiff", 1e-9))
		join_rows(X(idx_left), X(idx_right)).print("Y (1-3):");
	if (not approx_equal(Y(idx_left), Y(idx_right), "absdiff", 1e-9))
		join_rows(Y(idx_left), Y(idx_right)).print("Y (1-3):");
	D_interface = block_diag({kron(inv_rho, Dtheta), Icart, kron(Dx, Iy), Ipolar});
	stitch_interface(A, B, idx_left, idx_right, D_interface);

	// Region 2 <-> Region 4
	// Note the sign when matching x derivatives to theta derivatives
	idx_left  = n2 + idx_x_max.subvec(1, n_y-2);
	idx_right = n4 + idx_theta_max.subvec(1, n_rho-2);
	if (not approx_equal(X(idx_left), X(idx_right), "absdiff", 1e-9))
		join_rows(X(idx_left), X(idx_right)).print("Y (2-4):");
	if (not approx_equal(Y(idx_left), Y(idx_right), "absdiff", 1e-9))
		join_rows(Y(idx_left), Y(idx_right)).print("Y (2-4):");
	D_interface = block_diag({Ipolar, kron(Dx, Iy), Icart, kron(inv_rho, -Dtheta)});
	stitch_interface(A, B, idx_left, idx_right, D_interface);

	// Region 3 <-> Region 4
	idx_left  = n3 + reverse(idx_x_max.subvec(1, n_y-2));
	idx_right = n4 + idx_theta_min.subvec(1, n_rho-2);
	if (not approx_equal(X(idx_left), X(idx_right), "absdiff", 1e-9))
		join_rows(X(idx_left), X(idx_right)).print("Y (3-4):");
	if (not approx_equal(Y(idx_left), Y(idx_right), "absdiff", 1e-9))
		join_rows(Y(idx_left), Y(idx_right)).print("Y (3-4):");
	D_interface = block_diag({Ipolar, Icart, kron(Dx, Iy), kron(inv_rho, Dtheta)});
	stitch_interface(A, B, idx_left, idx_right, D_interface);

	dirichlet_bc(A, B, idx_boundary_points);

	std::cout << "H size: " << A.n_rows << " x " << A.n_cols << std::endl;
	std::cout << "H has NaN: " << A.has_nan() << std::endl;
	std::cout << "H has Inf: " << A.has_inf() << std::endl;

	A.save("A.csv", csv_ascii);
	B.save("B.csv", csv_ascii);
	const auto [eigval, eigvec] = diagonalize_pair(A, B);
	
	////////////////////
	// Build results
	////////////////////
	Results r;
	r.geometry = "stadium";
	r.x = X;
	r.y = Y;
	r.eigval = eigval;
	r.eigvec = eigvec;
	r.excluded_points = zeros(r.x.n_rows);
	return r;
}
