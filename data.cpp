#include "data.hpp"

shared_ptr<Data> build_data(const string &data_type, const pt::ptree &options,
                            const string &fname) {
  if (data_type == "dense") {
    return shared_ptr<Data>(new DenseData(options, fname));
  } else {
    throw runtime_error("unknown data type");
  }
}

DenseData::DenseData(const pt::ptree &options, const string &fname) {
  ifstream fin(fname);
  arma::uword n_rows, n_cols;
  fin >> n_rows >> n_cols;
  arma::mat tmp_data(n_rows, n_cols);
  float datum;
  for (arma::uword i = 0; i < n_rows; ++i) {
    for (arma::uword j = 0; j < n_cols; ++j) {
      fin >> datum;
      tmp_data(i, j) = datum;
    }
  }
  data = shared_ptr<arma::mat>(new arma::mat(tmp_data));
}

shared_ptr<Data> DenseData::transpose() const {
  DenseData *trans_data = new DenseData();
  trans_data->options = options;
  trans_data->data.reset(new arma::mat(data->t()));
  return shared_ptr<Data>(trans_data);
}
