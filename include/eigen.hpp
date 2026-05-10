#pragma once
#include <armadillo>

std::tuple<arma::cx_vec, arma::cx_mat> diagonalize(const arma::mat &H);
