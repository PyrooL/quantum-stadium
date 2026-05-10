#include "results.hpp"

void write_results(const Results& r) {

	int n_points = r.excluded_points.n_rows;
	int n_bc_points = arma::sum(r.excluded_points);

	// reconstruct eigenvectors with boundary-condition rows inserted
	arma::cx_mat eigvec_full(
		r.eigvec.n_rows + n_bc_points, 
		r.eigvec.n_cols,
		arma::fill::zeros
	);

	int src_row = 0;
	for (int dst_row = 0; dst_row < n_points; dst_row++) {
		if (!r.excluded_points(dst_row)) {
			eigvec_full.row(dst_row) = r.eigvec.row(src_row);
			src_row++;
		}
	}

	std::cout << "Eigvec size: " << eigvec_full.n_rows << " x " << eigvec_full.n_cols << "\n";


	// save results to CSV
	arma::vec eigval_real = arma::real(r.eigval);
	arma::vec eigval_imag = arma::imag(r.eigval);
	arma::mat eigvec_real = arma::real(eigvec_full);
	arma::mat eigvec_imag = arma::imag(eigvec_full);

	r.x.save("results/" + r.geometry + "_x.csv", arma::csv_ascii);
	r.y.save("results/" + r.geometry + "_y.csv", arma::csv_ascii);
	eigval_real.save("results/" + r.geometry + "_eigval_real.csv", arma::csv_ascii);
	eigval_imag.save("results/" + r.geometry + "_eigval_imag.csv", arma::csv_ascii);
	eigvec_real.save("results/" + r.geometry + "_eigvec_real.csv", arma::csv_ascii);
	eigvec_imag.save("results/" + r.geometry + "_eigvec_imag.csv", arma::csv_ascii);
}
