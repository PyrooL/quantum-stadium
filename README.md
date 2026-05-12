```
sudo dnf install cmake openblas-devel lapack-devel arpack-devel SuperLU-devel
sudo dnf install armadillo
```

Build with `make`

Run:
```
./build/main [geometry] [M] [N]
```

`geometry`: "stadium" (default), "square", "semicircle", "circle". Circle and stadium not implemented yet.

`M`: defaults to 32.

`N`: defaults to M if not passed, or 30 if M not passed either. 

stadium: polar endcaps have M radial points, N angular points. Cartesian center region is two rectangles of 2M points along x, M points along y.

square: M points along x, N points along y.

semicircle: M points along rho, N points along theta from -π/2 to +π/2.

circle: M points along rho, N points along theta from 0 to 2π.
