#include <iostream>
#include "eigen.hpp"

std::tuple<arma::cx_vec, arma::cx_mat> diagonalize(const arma::mat &M) {
	arma::cx_vec eigval;
	arma::cx_mat eigvec;
	bool diag_success = arma::eig_gen(eigval, eigvec, M);
	if (not diag_success) {
        throw std::runtime_error("Diagonalization failed");
	}
	return {eigval, eigvec};
}


