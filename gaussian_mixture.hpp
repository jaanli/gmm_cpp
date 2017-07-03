#ifndef GAUSSIAN_MIXTURE_HPP
#define GAUSSIAN_MIXTURE_HPP

#include <cmath>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf.h>

#include "dirichlet.hpp"
#include "link_function.hpp"
#include "model.hpp"
#include "normal.hpp"
#include "utils.hpp"

class PGaussianMixture : public Model {
private:
  map<string, unique_ptr<Model>> distributions;
  size_t n_components;

public:
  PGaussianMixture(const pt::ptree &options) : Model(options) {
    n_components = options.get<int>("p.n_components");
    distributions["mixture_weight"].reset(new PDirichlet(options));
    for (auto k = 0; k < n_components; k++) {
      string component_name = "component_loc_" + to_string(k);
      distributions[component_name].reset(
          new PNormal(options, options.get<int>("data_dimension")));
      distributions["likelihood"].reset(
          new PNormal(options, options.get<int>("data_dimension")));
    }
  }

  double compute_log_p(MapOfMat z) {
    double res = 0;
    res += distributions["mixture_weight"]->compute_log_p(*z["mixture_weight"]);
    for (auto k = 0; k < n_components; k++) {
      string component_name = "component_loc_" + to_string(k);
      res += distributions[component_name]->compute_log_p(*z[component_name]);
    }
    return res;
  };

  double compute_log_lik(arma::vec x, MapOfMat z) {
    double res = 0;
    for (auto k = 0; k < n_components; k++) {
      string component_name = "component_loc_" + to_string(k);
      auto loc = z[component_name];
      auto component_weight = (*z["mixture_weight"])(k);
      res += component_weight *
             distributions["likelihood"]->compute_log_p(x, *loc);
    }
    return res;
  };
};

class QGaussianMixture : public Variational {
protected:
  size_t n_components;

public:
  QGaussianMixture(const pt::ptree &options) : Variational(options) {
    n_components = options.get<int>("p.n_components");
    distributions["mixture_weight"] =
        unique_ptr<QDirichlet>(new QDirichlet(options));
    for (auto k = 0; k < n_components; k++) {
      string component_name = "component_loc_" + to_string(k);
      distributions[component_name] = unique_ptr<QNormal>(
          new QNormal(options, options.get<int>("data_dimension")));
    }
  }

  void print() {
    for (const auto &p : distributions) {
      cout << p.first << ": " << endl;
      distributions[p.first]->print();
    }
  }

  MapOfMat samples(gsl_rng *rng) {
    MapOfMat z;
    for (const auto &p : distributions)
      z[p.first] = distributions[p.first]->sample(rng);
    return z;
  };

  double compute_log_q(MapOfMat z) {
    double res = 0;
    for (const auto &p : distributions)
      res += distributions[p.first]->compute_log_q(*z[p.first]);
    return res;
  };

  MapOfMat grad_lq_matrix(MapOfMat z) {
    MapOfMat res;
    for (const auto &p : distributions)
      res[p.first] = distributions[p.first]->grad_lq_matrix(z[p.first]);
    return res;
  };
};

#endif
