#pragma once

#include "utils.hpp"

// a general data base class
class Data {
public:
  // sp_mat || mat
  virtual string get_data_type() = 0;
  virtual shared_ptr<arma::sp_mat> get_sp_mat() {
    throw runtime_error("get_sp_mat() not implemented in Data");
    return NULL;
  }
  virtual shared_ptr<arma::mat> get_mat() {
    throw runtime_error("get_mat() not implemented in Data");
    return NULL;
  }

  // returns NULL by ault, not NULL means there is a
  // training/testing split
  virtual shared_ptr<arma::mat> get_train_filter() { return NULL; }

  virtual int n_examples() = 0;
  virtual int n_dim_y() = 0;
  virtual shared_ptr<Data> transpose() const = 0;
  virtual shared_ptr<arma::mat> slice_data(const ExampleIds &example_ids) = 0;

  virtual void transform(function<double(double)> func) = 0;
};

shared_ptr<Data> build_data(const string &data_type, const pt::ptree &options,
                            const string &fname);

class DenseData : public Data {
private:
  pt::ptree options;
  shared_ptr<arma::mat> data;

  DenseData() {}

public:
  string get_data_type() { return "mat"; }
  shared_ptr<arma::mat> get_mat() { return data; }

  int n_examples() { return data->n_cols; }

  int n_dim_y() { return data->n_rows; }

  shared_ptr<Data> transpose() const;

  DenseData(const pt::ptree &options, const string &fname);

  shared_ptr<arma::mat> slice_data(const ExampleIds &example_ids) {
    shared_ptr<arma::mat> batch(
        new arma::mat(data->n_rows, example_ids.size()));
    for (size_t i = 0; i < example_ids.size(); ++i)
      batch->col(i) = data->col(example_ids[i]);
    return batch;
  }

  void transform(function<double(double)> func) { data->transform(func); }
};
