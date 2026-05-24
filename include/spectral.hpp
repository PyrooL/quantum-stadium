#pragma once
#include <armadillo>

constexpr double pi = 3.14159265358979323846;
arma::vec gauss_lobatto(int n, double ymin = -1.0, double ymax = 1.0);
arma::mat chebyshev_diff_matrix(int n, double ymin = -1.0, double ymax = 1.0);
arma::mat fourier_diff_matrix(int n, double ymin = 0., double ymax = 2*pi);

// linalg helpers
int kron_index(int N_x, int N_y, int x_i, int y_i);
std::tuple<arma::cx_vec, arma::cx_mat> diagonalize(const arma::mat &M);
std::tuple<arma::cx_vec, arma::cx_mat> diagonalize_pair(const arma::mat &A, const arma::mat &B);
arma::mat block_diag(const std::vector<arma::mat>& blocks);
void stitch_interface(arma::mat& A, arma::mat& B, const arma::uvec& idx1, const arma::uvec& idx2, const arma::mat& D_interface);
void dirichlet_bc(arma::mat& A, arma::mat& B, const arma::uvec& idx_boundary_points);
