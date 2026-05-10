#include <iostream>
#include <string>
#include <cmath>
#include <armadillo>
#include "spectral.hpp"
#include "eigen.hpp"
#include "stadium/geometry.hpp"
#include "stadium/operators.hpp"
#include "semicircle/semicircle.hpp"
#include "square/square.hpp"
#include "results.hpp"

int main_stadium(int M, int N);
int main_semicircle(int M, int N);
int main_square(int M, int N);

int main(int argc, char *argv[]) {
	std::string mode = "stadium";
	int M = 30; 
	int N = 30;

	if (argc > 1) {
		mode = argv[1]; 
	}

	if (argc > 2) {
		M = std::stoi(argv[2]);
		N = M;
	}

	if (argc > 3) {
		N = std::stoi(argv[3]);
	}

	std::cout << "Pseudospectral solution to particle in a " << mode << std::endl;
	std::cout << "M = " << M << ", N = " << N << std::endl;
	if (mode == "stadium") {
		main_stadium(M, N);
	} else if (mode == "semicircle") {
		main_semicircle(M, N);
	} else if (mode == "square") {
		main_square(M, N);
	} else {
		return 1;
	}
	
	std::cout << "That's all, folks!\n";
    return 0;
}

int main_stadium(int M, int N) { 
	double L = 1.;
	
	// (M, N) radial/angular grid on endcamps, (2M, M) cartesian grids in the center
	auto g = build_stadium(M, N, L);
	std::cout << "Built stadium\n";
	auto dv = build_derivatives(g);
	std::cout << "Built derivatives\n";
	auto h = build_hamiltonian(g);
	std::cout << "Built hamiltonian\n";
	
	return 0;
}

int main_semicircle(int M, int N) {
	auto results = solve_semicircle(M, N);
	write_results(results);
	return 0;
}

int main_square(int M, int N) {
	auto results = solve_square(M, N);
	write_results(results);
	return 0;
}

