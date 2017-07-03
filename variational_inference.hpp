#pragma once

#include "bbvi.hpp"
#include "data.hpp"
#include "model.hpp"
#include "random.hpp"
#include "utils.hpp"

class VariationalInference {
private:
  vector<GSLRandom *> vec_rng;
  int iteration, n_samples, n_params;
  shared_ptr<Data> data;

protected:
  pt::ptree options;
  arma::uword n_examples;
  ExampleIds all_examples;
  gsl_rng *rng;
  int threads;
  vector<Variational::ScoreFunction> score_funcs;
  vector<Serializable<arma::mat> *> param_matrices;
  Model *model;
  Variational *variational;

public:
  VariationalInference(pt::ptree &options, Model *p, Variational *q) {
    this->options = options;
    model = p;
    variational = q;
    init();
  }

  void init() {
    auto seed = options.get<int>("seed");
    n_samples = options.get<int>("samples");
    vec_rng.resize(n_samples);
    auto data_type = "dense";
    auto data_file = options.get<string>("data_file");
    data = build_data(data_type, options, data_file);
    n_examples = data->n_examples();
    for (int i = 0; i < n_samples; ++i) {
      vec_rng[i] = new GSLRandom();
      gsl_rng_set(vec_rng[i]->rng, seed + i);
    }
    iteration = 0;
    rng = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rng, seed);
    all_examples.clear();
    for (arma::uword j = 0; j < n_examples; ++j)
      all_examples.push_back(j);
    threads = options.get<int>("n_threads");
  }

  struct TrainStats {
    int iteration;
    arma::vec elbo;

    vector<arma::vec> lp_z;
    vector<arma::vec> lq_z;
    vector<BBVIStats> bbvi_stats_z;

    TrainStats(int iteration, int samples)
        : iteration(iteration), elbo(samples, arma::fill::zeros) {}
  };

  void print_stats(const TrainStats &);

  void train();

  /* TrainStats train_batch(const ExampleIds &example_ids); */
  TrainStats train_batch_global(const ExampleIds &example_ids);
};
