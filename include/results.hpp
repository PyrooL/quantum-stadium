#pragma once
#include <armadillo>

struct Results {
	std::string geometry;
	arma::vec x;
	arma::vec y;
	arma::cx_vec eigval;
	arma::cx_mat eigvec;
	arma::vec excluded_points;
};

void write_results(const Results& r);
