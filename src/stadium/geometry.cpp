#include "spectral.hpp"
#include "stadium/geometry.hpp"
#include <cmath>

StadiumGeometry build_stadium(int M, int N, double L)
{
    StadiumGeometry g;
    g.M = M;
    g.N = N;
    g.L = L;

    // radial grid: M points from 0 to L/2
    g.r = gauss_lobatto(M, 0.0, 0.5 * L);

    // angular grids: N points forming semicircles
    g.theta_left  = gauss_lobatto(N, pi/2, 3*pi/2);
    g.theta_right = gauss_lobatto(N, -pi/2, pi/2);

    // Upper and lower cartesian grids 2*M points -L/2 <= x <= L/2 and 0 < |y| < L/2
    g.x = gauss_lobatto(2*M, -0.5 * L, 0.5 * L);
    g.y_upper = gauss_lobatto(M, 0.0, 0.5 * L);
    g.y_lower = gauss_lobatto(M, -0.5 * L, 0.0);

    return g;
}
