#pragma once
#include <armadillo>

struct StadiumGeometry {
    int M, N;
    double L;

    // radial (endcaps)
    arma::vec r;
    arma::vec theta_left;
    arma::vec theta_right;

    // rectangular core
    arma::vec x;
    arma::vec y_upper;
    arma::vec y_lower;
};

StadiumGeometry build_stadium(int M, int N, double L);
