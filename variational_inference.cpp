#include "variational_inference.hpp"

VariationalInference::TrainStats
VariationalInference::train_batch_global(const ExampleIds &example_ids) {
  auto samples = n_samples;
  TrainStats stats(iteration++, samples);

  vector<MapOfMat> z_samples;
  z_samples.resize(samples);

  VecOfMat samples_log_p, samples_log_q;
  samples_log_p.resize(samples);
  samples_log_q.resize(samples);

  MapVecOfMat samples_score_q;

  for (int s = 0; s < samples; ++s) {
    gsl_rng *rng = vec_rng[s]->rng;

    z_samples[s] = variational->samples(rng);

    if (s == 0) {
      for (const auto &p : z_samples[s])
        samples_score_q[p.first].resize(samples);
    }
    auto sample_score_q = variational->grad_lq_matrix(z_samples[s]);

    samples_log_p[s] = model->log_p_matrix(z_samples[s]);

    samples_log_q[s] = variational->log_q_matrix(z_samples[s]);

    for (const auto &p : z_samples[s]) {
      samples_score_q[p.first][s] = sample_score_q[p.first];
    }

    auto sampling_ratio = -1.0;
    sampling_ratio = (example_ids.size() + 0.0) / n_examples;

    // renormalize
    *samples_log_p[s] *= sampling_ratio;
    *samples_log_q[s] *= sampling_ratio;

    // compute log-likelihood of the data
    if (options.get<bool>("observations")) {
      for (int i = 0; i < example_ids.size(); ++i) {
        *samples_log_p[s] +=
            model->compute_log_lik(data->slice_data(example_ids), z_samples[s]);
      }
    }

    stats.elbo(s) += arma::accu(*samples_log_p[s]);
    stats.elbo(s) -= arma::accu(*samples_log_q[s]);
  }

  variational->update(samples_score_q, samples_log_p, samples_log_q);

  return stats;
}

void VariationalInference::print_stats(const TrainStats &stats) {
  printf("Iteration %d, ELBO %.3e, std %.3e\n", stats.iteration,
         arma::mean(stats.elbo), arma::stddev(stats.elbo));
}

static ExampleIds gen_example_ids(gsl_rng *rng, const string &batch_order,
                                  int batch_size, int n_examples,
                                  int *batch_st) {

  (*batch_st) %= n_examples;
  ExampleIds examples;
  for (int j = 0; j < batch_size; ++j) {
    if (batch_order == "seq") {
      examples.push_back((*batch_st));
      ++(*batch_st);
      (*batch_st) %= n_examples;
    } else {
      (*batch_st) = gsl_rng_get(rng) % n_examples;
      examples.push_back(*batch_st);
    }
  }
  return examples;
}

void VariationalInference::train() {
  auto batch_size = options.get<int>("batch_size");
  ExampleIds ex = gen_example_ids(vec_rng[0]->rng, "seq", batch_size,
                                  n_examples, &batch_size);
  for (auto i = 0; i < options.get<int>("n_iterations"); i++) {
    /* auto train_stats = train_batch(ex); */
    auto train_stats = train_batch_global(ex);
    if (i % options.get<int>("print_every") == 0) {
      print_stats(train_stats);
      variational->print();
    }
  }
}
