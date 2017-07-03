#ifndef NORMAL_HPP
#define NORMAL_HPP

#include <cmath>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf.h>

#include "link_function.hpp"
#include "model.hpp"
#include "utils.hpp"

double normal_log_prob(arma::vec z, arma::vec loc, arma::vec scale) {
  z -= loc;
  auto sigma2 = scale * scale;
  return arma::accu(0.5 * log(2 * arma::datum::pi * sigma2) -
                    z * z / (2 * sigma2));
}

class PNormal : public Model {
private:
  arma::vec loc;
  arma::vec scale;

public:
  using Model::Model; // inherit base constructors
  PNormal(const pt::ptree &options, int dimension) : Model(options) {
    /* this->options = options; */
    loc = arma::vec(dimension, arma::fill::zeros);
    scale = arma::vec(dimension, arma::fill::zeros);
    scale.fill(options.get<double>("p.init_scale"));
  }
  double compute_log_p(arma::vec z) { return normal_log_prob(z, loc, scale); }
  double compute_log_p(arma::vec z, arma::vec loc) {
    return normal_log_prob(z, loc, scale);
  }
};

class QNormal : public Variational {
protected:
  Serializable<arma::mat> wloc;
  Serializable<arma::mat> wscale;

public:
  using Variational::Variational;
  QNormal(const pt::ptree &options, arma::uword dimension)
      : Variational(options) {
    /* this->options = options; */
    wloc = arma::mat(dimension, 1);
    wloc.fill(0.01);
    wscale = arma::mat(dimension, 1);
    wscale.fill(options.get<double>("q.init_scale"));
    ScoreFunctionGlobal score_loc = [=](arma::vec z, arma::uword i) {
      return (z(i) - wloc(i)) / (wscale(i) * wscale(i));
    };
    register_param(&wloc, score_loc, false);
  }

  void print() { cout << wloc << endl; }

  shared_ptr<arma::mat> sample(gsl_rng *rng) {
    shared_ptr<arma::mat> z(new arma::mat(wloc.n_cols, 1, arma::fill::zeros));
    for (auto i = 0; i < wloc.n_cols; i++)
      (*z)(i, 0) = gsl_ran_gaussian(rng, wscale(i)) + wloc(i);
    return z;
  }

  double compute_log_q(arma::vec z) { return normal_log_prob(z, wloc, wscale); }
};

#endif
