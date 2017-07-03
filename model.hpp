#ifndef MODELS_HPP
#define MODELS_HPP

#include <gsl/gsl_rng.h>

#include "bbvi.hpp"
#include "optimizer.hpp"
#include "utils.hpp"

class Model {
protected:
  const pt::ptree options;

public:
  Model() {}
  Model(const pt::ptree &options) : options(options) {}
  virtual double compute_log_p(arma::vec z){};
  virtual double compute_log_p(arma::vec, arma::mat){};
  virtual double compute_log_p(MapOfMat z){};
  virtual double compute_log_lik(shared_ptr<arma::mat> x,
                                 shared_ptr<arma::mat> z){};
  // global variables
  virtual double compute_log_lik(shared_ptr<arma::mat> x, MapOfMat z){};

  shared_ptr<arma::mat> log_p_matrix(shared_ptr<arma::mat> z) {
    shared_ptr<arma::rowvec> log_p(new arma::rowvec(z->n_cols));
    for (arma::uword j = 0; j < z->n_cols; ++j) {
      (*log_p)(j) = compute_log_p((*z).col(j));
    }
    return log_p;
  }

  shared_ptr<arma::mat> log_p_matrix(MapOfMat z) {
    shared_ptr<arma::mat> log_p(new arma::mat(1, 1));
    (*log_p)(0, 0) = compute_log_p(z);
    return log_p;
  }
};

class Variational {
protected:
  const pt::ptree options;
  map<string, unique_ptr<Variational>> distributions;

public:
  Variational() {}
  Variational(const pt::ptree &options) : options(options) {}
  typedef function<double(arma::vec, arma::uword, arma::uword)> ScoreFunction;
  typedef function<double(arma::vec, arma::uword)> ScoreFunctionGlobal;
  vector<ScoreFunction> score_funcs;
  vector<ScoreFunctionGlobal> score_funcs_global;
  vector<Serializable<arma::mat> *> param_matrices;
  vector<arma::uword> sample_shape;
  vector<Optimizer> optimizers;

  virtual arma::mat sample(gsl_rng *rng, arma::uword j){};
  virtual shared_ptr<arma::mat> sample(gsl_rng *rng){};
  virtual MapOfMat samples(gsl_rng *rng){};
  virtual double sample(gsl_rng *rng, arma::uword i, arma::uword j){};
  virtual void print(){};

  shared_ptr<arma::mat> sample_matrix(gsl_rng *rng,
                                      const ExampleIds &example_ids) {
    auto n_rows = param_matrices[0]->n_rows;
    // sample matrix of shape [z_dim, batch_size]
    shared_ptr<arma::mat> sample_mat(new arma::mat(n_rows, example_ids.size()));
    arma::uword example_ind = 0;
    for (auto j : example_ids) {
      if (sample_shape[0] == 1) {
        for (arma::uword i = 0; i < n_rows; ++i) {
          (*sample_mat)(i, example_ind) = sample(rng, i, j);
        }
      } else {
        (*sample_mat).col(example_ind) = sample(rng, j);
      }
      example_ind++;
    }
    return sample_mat;
  }

  virtual double compute_log_q(arma::vec z, arma::uword i){};
  virtual double compute_log_q(arma::vec z){};
  virtual double compute_log_q(MapOfMat){};

  void register_param(Serializable<arma::mat> *param_mat,
                      ScoreFunctionGlobal score_func, bool deserialize) {
    if (!deserialize) {
      optimizers.emplace_back(options, param_mat);
      param_matrices.push_back(param_mat);
    }
    score_funcs_global.push_back(score_func);
  }

  shared_ptr<arma::mat> log_q_matrix(shared_ptr<arma::mat> z,
                                     const ExampleIds &example_ids) {
    shared_ptr<arma::mat> log_q(new arma::mat(z->n_cols, 1));
    arma::uword ind = 0;
    for (auto j : example_ids) {
      (*log_q)(ind, 1) = compute_log_q((*z).col(ind), j);
      ++ind;
    }
    return log_q;
  }

  shared_ptr<arma::mat> log_q_matrix(MapOfMat z) {
    shared_ptr<arma::mat> log_q(new arma::mat(1, 1));
    (*log_q)(0, 0) = compute_log_q(z);
    return log_q;
  }

  /* shared_ptr<arma::cube> grad_lq_matrix(shared_ptr<arma::mat> z, */
  /*                                       const ExampleIds &example_ids) { */
  /*   size_t n_params = param_matrices.size(); */
  /*   shared_ptr<arma::cube> grad_lp( */
  /*       new arma::cube(z->n_rows, z->n_cols, n_params)); */
  /*   arma::uword ind = 0; */
  /*   for (auto j : example_ids) { */
  /*     for (arma::uword i = 0; i < z->n_rows; ++i) { */
  /*       for (size_t k = 0; k < score_funcs.size(); ++k) { */
  /*         (*grad_lp)(i, ind, k) = score_funcs[k]((*z).col(ind), i, j); */
  /*       } */
  /*     } */
  /*     ++ind; */
  /*   } */
  /*   return grad_lp; */
  /* } */

  // only global latent variables, no per-datapoint latents.
  shared_ptr<arma::mat> grad_lq_matrix(shared_ptr<arma::mat> z) {
    size_t n_params = param_matrices.size();
    shared_ptr<arma::mat> grad_lq(new arma::mat(z->n_rows, n_params));
    for (arma::uword i = 0; i < z->n_rows; ++i) {
      for (size_t k = 0; k < score_funcs.size(); ++k) {
        (*grad_lq)(i, k) = score_funcs_global[k]((*z).col(0), i);
      }
    }
    return grad_lq;
  }

  // local latent variable model
  BBVIStats update(const VecOfMat &score_q, const VecOfMat &log_p,
                   const VecOfMat &log_q) {
    BBVIStats stats;
    auto n_params = param_matrices.size();
    for (arma::uword k = 0; k < n_params; ++k) {
      // This is inefficent just pass and index to bbvi
      BBVIStats stats_k;
      auto grad_k =
          grad_bbvi_factorized(options, score_q, log_p, log_q, stats_k,
                               options.get<int>("n_threads"));
      stats += stats_k;
      optimizers[k].update(*grad_k);
    }
    stats /= (n_params + 0.0);
    return stats;
  }

  // hierarchical model; needs a map of score functions and optimizers
  BBVIStats update(const MapVecOfMat &score_q, const VecOfMat &log_p,
                   const VecOfMat &log_q) {
    BBVIStats stats;
    for (const auto &element : score_q) {
      auto name = element.first;
      auto score = element.second;
      auto stats_ = distributions[name]->update(score, log_p, log_q);
      stats += stats_;
    }
    return stats;
  }

  // global latent variables for a hierarchical model
  virtual MapOfMat grad_lq_matrix(MapOfMat){};

  friend class Optimizer;
  friend class VariationalInference;
};

#endif
