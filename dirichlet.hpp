#ifndef DIRICHLET_HPP
#define DIRICHLET_HPP

#include <cmath>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf.h>

#include "link_function.hpp"
#include "model.hpp"
#include "utils.hpp"

class PDirichlet : public Model {
private:
  arma::vec alpha;

public:
  PDirichlet(){};

  PDirichlet(const pt::ptree &options) : Model(options) {
    /* this->options = options; */
    alpha = arma::vec(options.get<int>("p.n_components"), arma::fill::ones);
    alpha = alpha * options.get<double>("p.init_alpha");
  }

  double compute_log_p(arma::vec z) {
    double *alpha_ = alpha.memptr();
    double *z_ = z.memptr();
    size_t K = alpha.size();
    return gsl_ran_dirichlet_lnpdf(K, alpha_, z_);
  }
};

class QDirichlet : public Variational {
protected:
  Serializable<arma::mat> walpha;
  size_t n_components;
  LinkFunction *lf;

public:
  QDirichlet(){};

  QDirichlet(const pt::ptree &options) : Variational(options) {
    /* this->options = options; */
    n_components = options.get<arma::uword>("p.n_components");
    lf = get_link_function(options.get<string>("q.link_function"));
    walpha = arma::mat(n_components, 1, arma::fill::ones);
    walpha.transform([&](double val) {
      return val * lf->f_inv(options.get<double>("q.init_alpha"));
    });
    sample_shape = {walpha.n_rows};
    ScoreFunctionGlobal score_alpha = [=](arma::vec z, arma::uword i) {
      auto alpha = walpha.transform([&](double val) { return lf->f(val); });
      auto lf_g = lf->g(alpha(i));
      return lf_g *
             (gsl_sf_psi(alpha(i)) - gsl_sf_psi(arma::accu(alpha)) + log(z(i)));
    };
    register_param(&walpha, score_alpha, false);
  }

  void print() {
    cout << walpha.transform([&](double val) { return lf->f(val); }) << endl;
  };

  shared_ptr<arma::mat> sample(gsl_rng *rng) {
    arma::vec walpha_col = walpha.col(0);
    arma::vec alpha_col =
        walpha_col.transform([&](double val) { return lf->f(val); });
    double arr[n_components];
    double *alpha_col_ = alpha_col.memptr();
    gsl_ran_dirichlet(rng, n_components, alpha_col_, arr);
    vector<double> std_vec(arr, arr + n_components);
    // need to convert vector to mat of [n, 1] shape
    shared_ptr<arma::mat> res(new arma::mat(n_components, 1));
    *res = arma::conv_to<arma::mat>::from(std_vec);
    return res;
  }

  double compute_log_q(arma::vec z) {
    arma::vec wcol = walpha.col(0);
    arma::vec col = wcol.transform([&](double val) { return lf->f(val); });
    double *alpha_ = col.memptr();
    double *z_ = z.memptr();
    return gsl_ran_dirichlet_lnpdf(n_components, alpha_, z_);
  }

  arma::vec alpha() {
    return walpha.transform([&](double val) { return lf->f(val); });
  };
};

#endif
