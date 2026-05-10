#pragma once
#include <armadillo>
#include "geometry.hpp"

struct DerivativeMatrices {
	// First derivatives
    arma::mat D_r;
    arma::mat D_theta_left;
    arma::mat D_theta_right;
    arma::mat D_x;
    arma::mat D_y_upper;
    arma::mat D_y_lower;

	//Second derivatives
    arma::mat D2_r;
    arma::mat D2_theta_left;
    arma::mat D2_theta_right;
    arma::mat D2_x;
    arma::mat D2_y_upper;
    arma::mat D2_y_lower;

	// Laplacians on each domain
	arma::mat Lap_left;
	arma::mat Lap_right;
    arma::mat Lap_upper;
    arma::mat Lap_lower;
};

struct Hamiltonian {
	arma::mat h_left;
	arma::mat h_right;
	arma::mat h_upper;
	arma::mat h_lower;
};

DerivativeMatrices build_derivatives(const StadiumGeometry& g);
Hamiltonian build_hamiltonian(const StadiumGeometry& g);
