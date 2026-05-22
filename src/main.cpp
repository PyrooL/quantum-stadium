#include <iostream>
#include <string>
#include <cmath>
#include <armadillo>
#include "spectral.hpp"
#include "stadium/stadium.hpp"
#include "stadium/geometry.hpp"
#include "stadium/operators.hpp"
#include "semicircle/semicircle.hpp"
#include "circle/circle.hpp"
#include "square/square.hpp"
#include "square/stitched_square.hpp"
#include "results.hpp"

int main_stadium(int M, int N);
int main_semicircle(int M, int N);
int main_square(int M, int N);

int main(int argc, char *argv[]) {
	std::string mode = "stadium";
	int M = 20; 
	int N = 20;

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
	Results results;
	if (mode == "stadium") {
		results = solve_stadium(M, N);
	} else if (mode == "semicircle") {
		results = solve_semicircle(M, N);
	} else if (mode == "square") {
		// results = solve_square(M, N);
		results = solve_stitched_square(M);
	} else if (mode == "circle") {
		results = solve_circle(M, N);
	} else {
		return 1;
	}

	std::cout << results.geometry << std::endl;
	write_results(results);
	
	std::cout << "That's all, folks!\n";
    return 0;
}
