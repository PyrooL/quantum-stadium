#include <iostream>
#include <cmath>

#include "spectral.hpp"
#include "results.hpp"

Results solve_stadium(int M, int N) {
	using namespace arma;
	
	////////////////////
	// Define geometry
	////////////////////
	const vec rho = gauss_lobatto(M, 0., 1.);
	const vec theta = gauss_lobatto(N, 0, pi);
	const vec x = gauss_lobatto(2*M, -1., 0.);
	const vec y1 = gauss_lobatto(M, 0., 1.);
	const vec y2 = gauss_lobatto(M, -1., 0.);

	const unsigned int n_rho = rho.n_elem; // M+1
	const unsigned int n_theta = theta.n_elem; // N+1
	const unsigned int n_x = x.n_elem; // 2M+1
	const unsigned int n_y = y1.n_elem; // M+1

}
