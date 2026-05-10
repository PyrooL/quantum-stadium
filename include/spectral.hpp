#pragma once
#include <armadillo>

constexpr double pi = 3.14159265358979323846;
arma::vec gauss_lobatto(int n, double ymin = -1.0, double ymax = 1.0);
arma::mat chebyshev_diff_matrix(int n, double ymin = -1.0, double ymax = 1.0);
arma::mat fourier_diff_matrix(int n, double ymin = 0., double ymax = 2*pi);

int kron_index(int N_x, int N_y, int x_i, int y_i);
